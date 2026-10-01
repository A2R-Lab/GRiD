"""CUDA equivalence test for the generated time-integrator kernels.

Validates `grid::integrator<EULER|SEMI_IMPLICIT_EULER>` and the matching
`integrator_gradient<...>` / `integrator_with_gradient<...>` host
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
import dataclasses
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
    _thread_counts,
)
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_integrator_smoke_runner.cu")

# Both fixed- and floating-base emit value + gradient + both-at-once kernels for
# the EULER / SI-Euler / Midpoint / TRAPEZOIDAL / RK4 integrators (the floating SI-Euler /
# Midpoint / TRAPEZOIDAL / RK4 gradients carry the SE(3) dIntegrate chain-rule wiring);
# CONSTANT_ACCELERATION uses one evaluation on both bases. Floating-base MIMIC robots
# emit the gradient too (B3 RESOLVED 2026-06-02 — the floating multi-stage mimic
# gradient composes the correct B1 floating-mimic FD gradient in reduced tangent
# space and is structurally exact, matched by go2-floating non-mimic to ~3e-7).
# Every cell compares every sample and dt entrywise (audit 2026-09-30: the old
# floating-mimic scopes hid generator bugs, docs/agent_debugging_guide.md 7.z32/7.z34).
# Floating mimic cells may fall back to a full-matrix norm bound: fr3-floating's
# float32 rounding through the floating Minv misses single entries at dt=0.1
# (see _FLOATING_MIMIC_NORM_RTOL).
# (prefix, python-side integrator name, has_gradient, fixed_base_only)
# CONSTANT_ACCELERATION is single-stage; both fixed- and floating-base now emit value +
# gradient (the floating constant_acceleration gradient carries the SE(3) dIntegrate
# chain-rule wiring at the combined tangent w = dt*qd + 0.5*dt^2*qdd, mirroring
# the floating SI-Euler gradient). fixed_base_only is now False for all rows.
_INTEGRATORS = (
    ("integrator_euler",       "euler",                True,  False),
    ("integrator_si_euler",    "semi_implicit_euler",  True,  False),
    ("integrator_midpoint",    "midpoint",             True,  False),
    ("integrator_trapezoidal",         "trapezoidal",                  True,  False),
    ("integrator_rk4",         "rk4",                  True,  False),
    ("integrator_constant_acceleration", "constant_acceleration",          True,  False),
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
    # (the recurring mimic-overflow bug class).
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


def _torque_driven(project_model, sample):
    """The sample with its third vector replaced by the torque that produces its qdd.

    ``DynamicsSample.qdd`` is an acceleration. Fed in directly as the control torque,
    a torque of up to 50 on a light distal link gives a first-stage acceleration near
    1e5, and a dt=0.1 explicit step then diverges through its stages (G1 RK4 at
    high_acceleration: stage-4 qdd ~2e22, x_kp1 ~4e20). No float32 kernel can track
    the float64 reference through that: relative noise of one float32 ulp on each
    forward-dynamics evaluation already moves the result past rtol, so the comparison
    measured the step's conditioning rather than the kernel. u = ID(q, qd, qdd) makes
    the first-stage acceleration exactly the sample's qdd, so "high_acceleration"
    means 50 rad/s^2 as intended. A CPU noise model then keeps every value cell
    within float32 reach; a few dt=0.1 RK4 gradient cells on floating robots stay
    ill-conditioned (test/test_integrator_sample_conditioning.py covers values).
    """
    u = np.asarray(
        project_model.inverse_dynamics(sample.q, sample.qd, sample.qdd), dtype=np.float64,
    ).reshape(-1)
    return dataclasses.replace(sample, qdd=u)


# Floating mimic cells (fr3, h1_2): an entry that misses the scaled check may pass on
# the full-matrix norm instead. Audit 2026-09-30 with every sample and dt, gradients
# built on both robots: h1_2-floating passes entrywise everywhere (worst 0.10 of the
# allowance, RK4 at dt=0.1 included); fr3-floating misses single entries by up to
# 1.65x at high_acceleration, dt=0.1, with worst norm_rel 7.4e-4 -- the same float32
# floating-Minv rounding as the FD-gradient guards in cuda_harness. The old 1e-2
# guard, the value-only h1_2 cell and the zero/conservative, dt<=0.01 scopes are gone.
_FLOATING_MIMIC_NORM_RTOL = 3e-3


def _generate_header(project_model, build_dir: Path) -> Path:
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=False,
        FILE_NAMESPACE="grid",
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(output_path=str(header), codegen_profile="integrators")
    return header


def _compile_runner(build_dir: Path, tier: str | None = None):
    arch = _detect_cuda_arch()
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    flags = ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    # Tier override so the suite exercises the LITE/MINIMAL SPILL path (the
    # integrator gradient's FD-grad inner s_temp / s_D_qdd_stage band routes to
    # d_workspace / d_temp_spill) in addition to PERF. The math is tier-independent,
    # so a tier sweep must still match the reference. The default `tier` arg comes
    # from the parametrized `tier` fixture (TIER_SHARED + TIER_LITE); the legacy
    # GRID_CUDA_INTEGRATOR_TIER env still overrides it for ad-hoc single-tier runs.
    tier = os.environ.get("GRID_CUDA_INTEGRATOR_TIER", tier)
    if tier and tier != "TIER_SHARED":
        if tier not in ("TIER_SHARED", "TIER_LITE", "TIER_MINIMAL"):
            pytest.fail("GRID_CUDA_INTEGRATOR_TIER must be TIER_SHARED, TIER_LITE, or TIER_MINIMAL.")
        flags.append(f"-DGRID_DEFAULT_RESOURCE_TIER={tier}")
    return cached_nvcc_executable(
        [RUNNER_SOURCE, build_dir / "grid.cuh"], flags,
        exe_name="cuda_integrator_smoke_runner.exe", fallback_dir=build_dir,
        include_dirs=[glass_inc], what="CUDA integrator smoke runner",
    )


def _build_case(project_model, tmp_path, label, tier=None):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    return _compile_runner(build_dir, tier=tier)


def _sample_stdin_with_dt(sample, dt: float) -> str:
    base = _sample_to_stdin(sample)
    return base + f" {dt}\n"


def _run_sample(executable, compile_cmd, sample, dt: float, num_threads=None):
    stdout = _run_runner(executable, _sample_stdin_with_dt(sample, dt), compile_cmd, num_threads=num_threads)
    return _parse_runner_output(stdout)


def _assert_close_scaled(actual, expected, rtol, atol, err_msg, norm_rtol=None):
    """assert_allclose with the absolute floor raised to rtol*max|expected|.

    A structurally-zero entry (e.g. a coupling term that vanishes at this
    operating point) carries float32 round-off ~ rtol*scale; comparing it with
    a fixed tiny atol trips at high velocity/dt even though the kernel is
    correct. Flooring atol at the array's overall scale lets "small relative to
    the matrix" count as close, while a genuine error stays O(scale) and fails.
    With norm_rtol, an entrywise miss still passes when
    ||actual - expected|| <= norm_rtol * ||expected|| (the flagship harness's
    norm-guard semantics)."""
    expected_arr = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected_arr))) if expected_arr.size else 0.0
    try:
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=max(atol, rtol * scale), err_msg=err_msg,
        )
    except AssertionError:
        if norm_rtol is None:
            raise
        diff = np.linalg.norm((np.asarray(actual, dtype=np.float64) - expected_arr).reshape(-1))
        rel = diff / max(float(np.linalg.norm(expected_arr.reshape(-1))), 1e-300)
        if rel > norm_rtol:
            raise AssertionError(f"{err_msg}: entrywise miss and norm-relative error {rel:.3e} > {norm_rtol:.3e}")


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
    all six integrators (including single-stage constant acceleration), at PERF
    (TIER_SHARED) AND at a spilled tier (TIER_LITE) so the big-robot (g1/h1_2)
    resource-tier SPILL path (FD-grad inner s_temp / s_D_qdd_stage band -> global
    d_workspace / d_temp_spill) is covered, not just the all-in-smem PERF arena.
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    floating_mimic = _robot_has_mimic(project_model) and base_mode == "floating"
    norm_rtol = _FLOATING_MIMIC_NORM_RTOL if floating_mimic else None
    label = f"{robot_id}_{base_mode}_{tier}_cuda_integrator"
    executable, compile_cmd = _build_case(project_model, tmp_path, label, tier=tier)
    samples = [_torque_driven(project_model, s) for s in _samples(project_model)]
    dts = _dts()
    nv = project_model.nv
    nq = project_model.nq

    rtol = 5e-4
    atol = 5e-4

    # Sweep block thread counts (one warp + multi-warp + a session-random count)
    # to catch thread-count-dependent races; the kernel is compiled once and the
    # thread count is passed to the runner via argv.
    for num_threads in _thread_counts():
      for dt in dts:
        for sample in samples:
            # The third vector of each sample is the control torque u (see
            # _torque_driven).
            actual = _run_sample(executable, compile_cmd, sample, dt, num_threads=num_threads)
            u = sample.qdd
            for prefix, integrator_type, has_gradient, fixed_base_only in _INTEGRATORS:
                # Retain the per-row scope flag for future restricted schemes.
                if fixed_base_only and base_mode != "fixed":
                    continue
                expected_x_kp1 = project_model.integrator(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type,
                )
                x_kp1_block = np.asarray(actual[prefix + "_x_kp1"], dtype=np.float64).reshape(-1)
                assert x_kp1_block.shape == (nq + nv,), (
                    f"{prefix} x_kp1 shape {x_kp1_block.shape} (expected {(nq + nv,)})"
                )
                _assert_close_scaled(
                    x_kp1_block, expected_x_kp1, rtol, atol, norm_rtol=norm_rtol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )

                if not has_gradient:
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
                    dAB_block, expected_dAB, rtol, atol, norm_rtol=norm_rtol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    x_kp1_with_block, expected_x_kp1, rtol, atol, norm_rtol=norm_rtol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1_with_dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    dAB_with_block, expected_dAB, rtol, atol, norm_rtol=norm_rtol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB_with_x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )


def _num_bodies(project_model):
    r = project_model.robot
    for attr in ("get_num_bodies", "get_num_links"):
        if hasattr(r, attr):
            return int(getattr(r, attr)())
    return project_model.nv  # fixed-base non-mimic fallback (NUM_BODIES == NUM_VEL)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_cuda_integrator_fext_matches_python_reference(tmp_path, monkeypatch):
    """Integrator with NONZERO external forces matches the Python reference.

    Gate for the f_ext threading through the integrator value + gradient: GRiD now
    passes d_f_ext into the integrator's FD inner (value) and the gradient's
    vaf/ID linearization, so a nonzero f_ext must shift x_kp1 AND [A|B] to match
    FD(q,qd,u, f_ext). The runner reads a body-major local-frame f_ext (opt-in via
    GRID_RUNNER_FEXT) into hd_data->d_f_ext; the host integrator wrapper reads it.

    Fixed-base iiwa14 (well-conditioned; also covers the new CONSTANT_ACCELERATION with
    f_ext). The no-fext path stays byte-identical (env unset) and is covered by
    test_cuda_integrator_matches_python_reference.
    """
    robot_id, base_mode = "iiwa14", "fixed"
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    nq, nv = project_model.nq, project_model.nv
    nb = _num_bodies(project_model)

    executable, compile_cmd = _build_case(
        project_model, tmp_path, f"{robot_id}_{base_mode}_fext_cuda_integrator", tier="TIER_SHARED"
    )
    samples = [_torque_driven(project_model, s) for s in _samples(project_model)]
    dts = _dts()

    # Deterministic NONZERO body-major local-frame f_ext ([angular; linear] per body),
    # same convention the f_ext 3-way equivalence test uses.
    rng = np.random.default_rng(20260617)
    f_ext = [rng.uniform(-3.0, 3.0, size=6) for _ in range(nb)]
    f_ext_flat = np.concatenate(f_ext).astype(np.float64)
    f_ext_str = " ".join(repr(float(x)) for x in f_ext_flat) + "\n"

    monkeypatch.setenv("GRID_RUNNER_FEXT", "1")
    rtol = 5e-4
    atol = 5e-4

    for dt in dts:
        for sample in samples:
            u = sample.qdd  # the control torque (see _torque_driven)
            stdin = _sample_stdin_with_dt(sample, dt) + f_ext_str
            actual = _parse_runner_output(_run_runner(executable, stdin, compile_cmd))
            # The runner must have actually received the f_ext we fed it.
            echoed = np.asarray(actual["input_f_ext"], dtype=np.float64).reshape(-1)
            np.testing.assert_allclose(
                echoed, f_ext_flat, rtol=0.0, atol=1e-5,
                err_msg="runner did not receive the f_ext sent on stdin",
            )
            for prefix, integrator_type, has_gradient, fixed_base_only in _INTEGRATORS:
                if fixed_base_only and base_mode != "fixed":
                    continue
                exp_x = project_model.integrator(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type, f_ext=f_ext,
                )
                x_blk = np.asarray(actual[prefix + "_x_kp1"], dtype=np.float64).reshape(-1)
                assert x_blk.shape == (nq + nv,), f"{prefix} x_kp1 shape {x_blk.shape}"
                _assert_close_scaled(
                    x_blk, exp_x, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} fext {prefix} x_kp1 @ {sample.name} dt={dt}",
                )
                if not has_gradient:
                    continue
                exp_dAB = project_model.integrator_gradient(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type, f_ext=f_ext,
                )
                dAB_blk = np.asarray(actual[prefix + "_dAB"], dtype=np.float64)
                assert dAB_blk.shape == (2 * nv, 3 * nv), f"{prefix} dAB shape {dAB_blk.shape}"
                _assert_close_scaled(
                    dAB_blk, exp_dAB, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} fext {prefix} dAB @ {sample.name} dt={dt}",
                )

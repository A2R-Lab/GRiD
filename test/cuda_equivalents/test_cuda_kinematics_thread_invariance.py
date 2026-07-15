"""V6 — kinematics thread-count-invariance matrix.

GRiD's single-block kernels are a CORE design invariant: every parallel region
is a block-stride loop (``for i = tid; i < N; i += blockDim.x``), so ANY block
size that fits MUST produce identical results (design_principles.rst §1). This
test pins that down for the KINEMATICS kernels specifically, sweeping the LOW end
of the thread range (1, 2, 16) that the broad sweep in
``test_cuda_executable_equivalence.py::_thread_counts`` (which only covers
{32, 96, 100, MAX_PERF}) does not exercise. The low end is the point: a missing
``__syncthreads`` between a block-stride write phase and a subsequent
read/accumulate phase is invisible at high occupancy / within a single warp but
diverges at 1, 2, or 16 threads.

Two assertions per (robot, base, thread_count, sample):
  1. ORACLE equivalence — CUDA matches the RBDReference (pinocchio-authoritative)
     numpy reference within the per-robot float32 tolerance bucket (reuses the
     executable-equivalence harness's ``_assert_close`` + tolerance tables).
  2. CROSS-THREAD invariance — the SAME input at thread_count N agrees with the
     thread_count=32 baseline to ~machine-float32 (rtol/atol = 1e-5), FAR tighter
     than the oracle tolerance. A divergence here is a thread-count-dependent race
     (a missing ``__syncthreads``), NOT a tolerance issue: identical inputs at the
     same precision must agree. This is the high-signal race detector.

Algorithms: end_effector_pose, end_effector_pose_gradient, frame_jacobian
(the current emitted symbols / algorithm_list keys — verified against the codegen).

Matrix (per algorithm):
  thread counts : {1, 2, 16, 32, 64, 128, 256}   (override GRID_CUDA_KIN_THREADS)
  batch sizes   : {1, 16, 256}  (= number of distinct input samples compared;
                   override GRID_CUDA_KIN_BATCHES)
  robots/bases  : iiwa14-fixed + go2-floating  (+ g1-floating for frame_jacobian)

Robots override via GRID_CUDA_KIN_ROBOTS="iiwa14:fixed,go2:floating".
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents import build_adapter, resolve_backend

from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _assert_close,
    _build_cuda_samples,
    _compile_runner,
    _ee_gimbal_lock_leaves,
    _expected_output,
    _generate_grid_header,
    _parse_runner_output,
    _run_runner,
    _sample_to_stdin,
)
from test.cuda_equivalents import test_cuda_frame_jacobian as fj


# Thread counts: the LOW end (1, 2, 16) catches missing __syncthreads between a
# block-stride write and a read/accumulate that only happens to be correct within
# a single warp / at high occupancy; the explicit powers of two extend the broad
# sweep's {32, 96, 100, MAX_PERF}. All are <= every target robot's
# MAX_PERF_LEVEL_THREADS (iiwa14=352, go2=288, g1=512), so the runner honors each
# verbatim (it only clamps counts that EXCEED the per-robot cap).
_DEFAULT_THREAD_COUNTS = (1, 2, 16, 32, 64, 128, 256)
# Batch sizes = number of distinct input samples driven through each cell. Single-
# block kernels are one-block-per-robot, so "batch" exercises the per-block compute
# at distinct configurations; thread invariance must hold for EVERY one.
_DEFAULT_BATCH_SIZES = (1, 16, 256)
_INVARIANCE_BASELINE_THREADS = 32
# Cross-thread invariance is a SAME-precision comparison: float32 CUDA at thread N
# vs float32 CUDA at thread 32. They must agree to ~machine-float32 round-off, far
# tighter than the float32-vs-float64 oracle tolerance. Any larger gap is a race.
_INVARIANCE_RTOL = 1e-5
_INVARIANCE_ATOL = 1e-5


def _thread_counts() -> tuple[int, ...]:
    raw = os.environ.get("GRID_CUDA_KIN_THREADS")
    if not raw:
        return _DEFAULT_THREAD_COUNTS
    counts = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    return tuple(dict.fromkeys(counts)) or _DEFAULT_THREAD_COUNTS


def _batch_sizes() -> tuple[int, ...]:
    raw = os.environ.get("GRID_CUDA_KIN_BATCHES")
    if not raw:
        return _DEFAULT_BATCH_SIZES
    sizes = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    return tuple(dict.fromkeys(sizes)) or _DEFAULT_BATCH_SIZES


def _ee_robot_modes():
    """Robots for the end_effector_pose / end_effector_pose_gradient invariance
    sweep: iiwa14-fixed + go2-floating at minimum."""
    raw = os.environ.get("GRID_CUDA_KIN_ROBOTS", "iiwa14:fixed,go2:floating")
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


def _frame_jac_robot_modes():
    """frame_jacobian invariance sweep adds a big robot (g1-floating) when the
    compile/runtime budget allows (override GRID_CUDA_KIN_FRAME_JAC_ROBOTS)."""
    raw = os.environ.get(
        "GRID_CUDA_KIN_FRAME_JAC_ROBOTS", "iiwa14:fixed,go2:floating,g1:floating"
    )
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


def _resolve_models(robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    oracle_backend = resolve_backend(os.environ.get("GRID_REFERENCE_BACKEND", "pinocchio"))
    reference_model = (
        project_model
        if oracle_backend == "reference"
        else build_adapter(spec, resolved, base_mode=base_mode, backend=oracle_backend)
    )
    return spec, resolved, project_model, reference_model


def _deterministic_samples(project_model, batch_size):
    """`batch_size` deterministic input samples (corner samples first, then
    seeded-random fillers). Reuses the executable-equivalence sample builder so
    the configs match the broad sweep's distribution; truncated/padded to exactly
    `batch_size` so the cell size is honored."""
    pool = _build_cuda_samples(
        project_model,
        random_count=max(batch_size, 4),
        include_corner_samples=True,
    )
    if len(pool) >= batch_size:
        return pool[:batch_size]
    # Pad by cycling (only if the pool is smaller than the requested batch).
    out = list(pool)
    i = 0
    while len(out) < batch_size:
        out.append(pool[i % len(pool)])
        i += 1
    return out


@contextlib.contextmanager
def _floating_emit_only(names):
    """Scope GRID_CUDA_FLOATING_ALGORITHMS so the FLOATING equivalence runner
    emits ONLY the requested kinematics kernels. Without this, an unset env makes
    the floating runner additionally emit the (heavy, workspace-dependent)
    end_effector_pose_hessian, which is not part of the V6 kinematics target and
    would only add launch risk / nvcc time. The runner reads this env at runtime
    in floating_algorithm_requested()."""
    key = "GRID_CUDA_FLOATING_ALGORITHMS"
    prev = os.environ.get(key)
    os.environ[key] = ",".join(names)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _assert_thread_invariant(baseline, candidate, label):
    """Same-input, different-thread-count outputs must agree to ~machine-float32.
    A larger gap is a thread-count-dependent race (missing __syncthreads), never a
    tolerance artifact (identical inputs at identical precision)."""
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(base) & np.isfinite(cand)
    # ⚠ Do NOT `return` on all-non-finite (2026-07-14 audit). Kinematics outputs (positions, Jacobians)
    # are finite for ANY finite q -- only orientation-derivative rows can legitimately go singular, and
    # never ALL of them. So both runs being entirely non-finite is a BROKEN kernel, not thread-invariance,
    # and silently returning declared it "invariant" while comparing nothing. Fail instead. (A PARTIAL
    # NaN still compares its finite part below, which is correct: consistent NaN at a singular row is fine.)
    assert np.any(finite), (
        f"{label}: BOTH thread-count runs are ENTIRELY non-finite. That is a broken kernel, not "
        f"thread-invariance -- a finite q must yield finite kinematics. Investigate the codegen output."
    )
    base_f = base[finite]
    cand_f = cand[finite]
    scale = float(np.max(np.abs(base_f))) if base_f.size else 0.0
    atol_eff = max(_INVARIANCE_ATOL, _INVARIANCE_RTOL * scale)
    diff = np.abs(cand_f - base_f)
    if np.all(diff <= atol_eff + _INVARIANCE_RTOL * np.abs(base_f)):
        return
    idx = int(np.argmax(diff))
    raise AssertionError(
        f"{label}: thread-count-dependent DIVERGENCE (missing __syncthreads?). "
        f"max_abs={diff[idx]:.3e} at flat#{idx}, "
        f"baseline(threads={_INVARIANCE_BASELINE_THREADS})={base_f[idx]:.9g}, "
        f"candidate={cand_f[idx]:.9g}, atol_eff={atol_eff:.3e} (scale={scale:.3e})"
    )


# ---------------------------------------------------------------------------
# end_effector_pose + end_effector_pose_gradient (main equivalence runner)
# ---------------------------------------------------------------------------
_EE_ALGORITHMS = ("end_effector_pose", "end_effector_pose_gradient")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("batch_size", _batch_sizes(), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize(("robot_id", "base_mode"), _ee_robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_ee_pose_thread_invariance(tmp_path, robot_id, base_mode, batch_size, request):
    if base_mode == "floating":
        request.node.add_marker(pytest.mark.floating_base)
    spec, resolved, project_model, reference_model = _resolve_models(robot_id, base_mode)

    build_dir = tmp_path / f"kin_ee_{robot_id}_{base_mode}_b{batch_size}"
    build_dir.mkdir()
    header_path, header_key = _generate_grid_header(
        project_model, resolved, build_dir, request.config
    )

    # For FLOATING base, restrict the runner to emit ONLY the two kinematics
    # kernels under test (the value + gradient), keeping the heavy ee_pose_hessian
    # out of the compile + launch. Fixed base emits the kinematics kernels by
    # default, no scoping needed.
    emit_scope = (
        _floating_emit_only(_EE_ALGORITHMS)
        if base_mode == "floating"
        else contextlib.nullcontext()
    )

    failures = []
    with emit_scope:
        executable, compile_cmd = _compile_runner(
            build_dir,
            floating_base=base_mode == "floating",
            header_key=header_key,
            skip_gradients=False,
            skip_eepose_gradients=False,
            config=request.config,
        )

        thread_counts = _thread_counts()
        n_leaves = len(project_model.robot.get_leaf_nodes())
        samples = _deterministic_samples(project_model, batch_size)

        failures = _run_ee_invariance_loop(
            samples, thread_counts, executable, compile_cmd, reference_model,
            project_model, robot_id, base_mode, n_leaves,
        )

    if failures:
        pytest.fail(
            f"{robot_id}-{base_mode} kinematics thread-invariance failures "
            f"(batch={batch_size}):\n" + "\n".join(failures)
        )


def _run_ee_invariance_loop(samples, thread_counts, executable, compile_cmd,
                            reference_model, project_model, robot_id, base_mode, n_leaves):
    failures = []
    for sample in samples:
        stdin = _sample_to_stdin(sample)
        gimbal = {
            name: _ee_gimbal_lock_leaves(reference_model, project_model, sample)
            for name in _EE_ALGORITHMS
        }
        expected = {}
        for name in _EE_ALGORITHMS:
            ev = _expected_output(reference_model, project_model, sample, name)
            expected[name] = np.asarray(ev, dtype=np.float64)
        baseline = {}
        for threads in thread_counts:
            cuda = _parse_runner_output(
                _run_runner(executable, stdin, compile_cmd, num_threads=threads)
            )
            for name in _EE_ALGORITHMS:
                if name not in cuda:
                    continue
                actual = cuda[name]
                tag = f"{robot_id}-{base_mode}/{sample.name}/{name}/threads={threads}"
                # (1) oracle equivalence — skip if the reference is undefined here
                # (e.g. rpy gimbal lock makes the orientation derivative non-finite).
                if np.all(np.isfinite(expected[name])):
                    try:
                        _assert_close(
                            tag, actual, expected[name],
                            robot_id=robot_id, algorithm=name,
                            n_leaves=n_leaves, gimbal_lock_leaves=gimbal[name],
                        )
                    except AssertionError as exc:
                        failures.append(str(exc))
                # (2) cross-thread invariance vs the 32-thread baseline.
                if threads == _INVARIANCE_BASELINE_THREADS:
                    baseline[name] = np.asarray(actual, dtype=np.float64)
                elif name in baseline:
                    try:
                        _assert_thread_invariant(baseline[name], actual, tag)
                    except AssertionError as exc:
                        failures.append(str(exc))
    return failures


# ---------------------------------------------------------------------------
# frame_jacobian (frame_jacobian smoke runner)
# ---------------------------------------------------------------------------
@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("batch_size", _batch_sizes(), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize(("robot_id", "base_mode"), _frame_jac_robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_frame_jacobian_thread_invariance(tmp_path, robot_id, base_mode, batch_size, request):
    if base_mode == "floating":
        request.node.add_marker(pytest.mark.floating_base)
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)

    build_dir = tmp_path / f"kin_fj_{robot_id}_{base_mode}_b{batch_size}"
    build_dir.mkdir()
    fj._generate_header(project_model, build_dir)
    executable, cmd = fj._compile_runner(build_dir)

    robot = project_model.robot
    leaf_id = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_id).get_name()
    nv = project_model.nv
    thread_counts = _thread_counts()
    samples = _deterministic_samples(project_model, batch_size)

    # frame_jacobian J only (the V6 target). Jd/Lambda are validated by the
    # dedicated test_cuda_frame_jacobian.py; here we pin J thread-invariance.
    _REF_FRAMES = (("J_local", "LOCAL"), ("J_world", "WORLD"), ("J_lwa", "LOCAL_WORLD_ALIGNED"))

    def close(actual, expected, msg, rtol=2e-3, atol=2e-3):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    failures = []
    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        qd = np.asarray(sample.qd, np.float64)
        stdin = fj._stdin(leaf_id, q, qd)
        J_ref = {
            ref_frame: np.asarray(
                project_model.frame_jacobian(q, leaf_name, ref_frame), dtype=np.float64
            )
            for _, ref_frame in _REF_FRAMES
        }
        baseline = {}
        for threads in thread_counts:
            out = _parse_runner_output(_run_runner(executable, stdin, cmd, num_threads=threads))
            for jblk, ref_frame in _REF_FRAMES:
                actual = out[jblk].reshape(6, nv, order="F")
                tag = f"{robot_id}-{base_mode}/{sample.name}/frame_jacobian/{ref_frame}/threads={threads}"
                # (1) oracle equivalence (analytic vs analytic, tight float32 bucket).
                try:
                    close(actual, J_ref[ref_frame], f"J {tag}")
                except AssertionError as exc:
                    failures.append(str(exc))
                # (2) cross-thread invariance vs the 32-thread baseline.
                if threads == _INVARIANCE_BASELINE_THREADS:
                    baseline[ref_frame] = np.asarray(actual, dtype=np.float64)
                elif ref_frame in baseline:
                    try:
                        _assert_thread_invariant(baseline[ref_frame], actual, tag)
                    except AssertionError as exc:
                        failures.append(str(exc))

    if failures:
        pytest.fail(
            f"{robot_id}-{base_mode} frame_jacobian thread-invariance failures "
            f"(batch={batch_size}):\n" + "\n".join(failures)
        )

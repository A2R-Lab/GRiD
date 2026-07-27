"""Host-surface CUDA equivalence test for the general-frame Jacobian family (S1).

Complements `test_cuda_frame_jacobian.py` (which drives the `*_device` functions
from a hand-written kernel) by exercising the BATCHED, launchable HOST surface
end-to-end through the gridData output buffers:

  * grid::frame_jacobian      -> hd_data->d_frame_jacobian / h_frame_jacobian
  * grid::frame_jacobian_dot  -> hd_data->d_frame_jacobian_dot / h_frame_jacobian_dot   (when emitted)
  * grid::osc_inertia         -> hd_data->d_osc_inertia / h_osc_inertia                 (when emitted)

The launchable surface bakes a fixed frame target (the leaf-EE joint) and the
LOCAL_WORLD_ALIGNED reference frame, so this test cross-checks ONLY that
(target, LWA) pair against the RBDReference numpy oracle (which matches pinocchio
to ~1e-14). float32 device -> float32-scale tolerance.

Robots: iiwa14-fixed + go2-floating (override with
GRID_CUDA_FRAME_JAC_HOST_ROBOTS="iiwa14:fixed,go2:floating").
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
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_frame_jacobian_host_runner.cu")
_ALGO_KEYS = ["frame_jacobian", "frame_jacobian_dot", "osc_inertia"]
_REF_FRAME = "LOCAL_WORLD_ALIGNED"  # the host's baked default reference frame


def _robot_modes():
    # fr3 (fixed + floating) is a MIMIC robot: its mimic joint shares its target's
    # reduced v-slot, so the frame Jacobian + its Jdot fold the mimic body's column
    # alpha-weighted into the shared column (frame_jacobian_dot_device's fjc_alpha /
    # fjd_alpha path). Covers the analytic-CUDA-vs-analytic-oracle mimic transcription.
    raw = os.environ.get("GRID_CUDA_FRAME_JAC_HOST_ROBOTS",
                         "iiwa14:fixed,go2:floating,fr3:fixed,fr3:floating")
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
        codegen.gen_all_code(algorithm_list=_ALGO_KEYS, output_path=str(header))
    return header


def _compile_runner(build_dir, defines):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_frame_jacobian_host_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ] + [f"-D{d}" for d in defines]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA frame_jacobian host runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(q, qd):
    rows = [" ".join(f"{v:.9g}" for v in np.asarray(q, dtype=np.float32)),
            " ".join(f"{v:.9g}" for v in np.asarray(qd, dtype=np.float32))]
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_frame_jacobian_host_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_frame_jac_host"
    build_dir.mkdir()
    header = _generate_header(project_model, build_dir)
    header_txt = header.read_text()
    # The host runner gates the dot/Lambda surfaces on codegen-emitted markers; pass
    # the matching -D so the runner only calls what the header actually defines.
    defines = []
    if "GRID_HAS_FRAME_JACOBIAN_DOT" in header_txt:
        defines.append("GRID_HAS_FRAME_JACOBIAN_DOT")
    if "GRID_HAS_OSC_INERTIA" in header_txt:
        defines.append("GRID_HAS_OSC_INERTIA")
    executable, cmd = _compile_runner(build_dir, defines)

    robot = project_model.robot
    leaf_id = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_id).get_name()
    nv = project_model.nv

    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    def close(actual, expected, msg, rtol=2e-3, atol=2e-3):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    rel_errs = []
    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        qd = np.asarray(sample.qd, np.float64)
        out = _parse_runner_output(_run_runner(executable, _stdin(q, qd), cmd))
        tag = f"{robot_id}-{base_mode} @ {sample.name} {_REF_FRAME} (host)"
        J_ref = np.asarray(
            project_model.frame_jacobian(q, leaf_name, _REF_FRAME), dtype=np.float64)
        J_cuda = out["FJ"].reshape(6, nv, order="F")
        close(J_cuda, J_ref, f"J {tag}")
        denom = max(float(np.max(np.abs(J_ref))), 1e-9)
        rel_errs.append(float(np.max(np.abs(J_cuda - J_ref))) / denom)

        if "FJD" in out:
            # Jdot is now ANALYTIC on BOTH sides (device + oracle) -> tight f32 tol,
            # matching the value J check (was 5e-2 for the old central-FD device).
            Jd_ref = np.asarray(
                project_model.frame_jacobian_dot(q, qd, leaf_name, _REF_FRAME),
                dtype=np.float64)
            close(out["FJD"].reshape(6, nv, order="F"), Jd_ref, f"Jdot {tag}",
                  rtol=2e-3, atol=2e-3)

        if "LAM" in out:
            task = (J_ref @ np.asarray(project_model.minv(q), dtype=np.float64) @ J_ref.T)
            if np.linalg.cond(task) < 1e8:
                L_ref = np.asarray(
                    project_model.osc_inertia(q, leaf_name, _REF_FRAME),
                    dtype=np.float64)
                close(out["LAM"].reshape(6, 6, order="F"), L_ref, f"Lambda {tag}",
                      rtol=5e-3, atol=5e-3)
    print(f"[frame_jacobian host] {robot_id}-{base_mode} max rel err (J, LWA): "
          f"{max(rel_errs):.2e}")

"""Batched host-wrapper CUDA equivalence test — regression guard for the floating
nq-vs-nv matrix-buffer stride bug class (project_grid_floating_nqnv_bug_class).

Every other CUDA equivalence test drives either the `*_device` functions or the
kernels with single-timestep (`init_gridData<T,1>`) buffers it owns itself, so none
exercise the gridData `h_*` host-wrapper copy at NUM_TIMESTEPS>1. That blind spot is
exactly where the matrix outputs (Minv, M, dc_du, df_du) hid a per-timestep stride
bug: the kernels write `nv*nv` but the host malloc/copy used `nq*nq`, so for a
FLOATING base (nq>nv) batch slot k>0 was corrupted (silent on fixed base, nq==nv,
and on batch=1, slot 0).

This test runs the BATCHED host wrappers `grid::minv` / `grid::crba` at
NUM_TIMESTEPS=GRID_BATCH on a floating robot with DISTINCT states per slot and
checks EVERY slot against the RBDReference oracle — so a per-timestep stride
regression fails loudly. Fixed-base is included as a control (must stay correct).
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
    _normalize_cuda_minv,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

RUNNER_SOURCE = Path(__file__).with_name("cuda_batched_host_runner.cu")
_BATCH = 3
_ALGO_KEYS = ["minv", "crba"]


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_BATCHED_HOST_ROBOTS", "iiwa14:fixed,go2:floating")
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
    executable = build_dir / "cuda_batched_host_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-DGRID_BATCH={_BATCH}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ] + [f"-D{d}" for d in defines]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA batched host runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(samples):
    rows = [" ".join(f"{v:.9g}" for v in np.asarray(s.q, dtype=np.float32)) for s in samples]
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_batched_host_wrapper_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_batched_host"
    build_dir.mkdir()
    header = _generate_header(project_model, build_dir)
    header_txt = header.read_text()
    defines = []
    if "GRID_HAS_CRBA" in header_txt or "crba(" in header_txt:
        defines.append("GRID_HAS_CRBA")
    executable, cmd = _compile_runner(build_dir, defines)

    nv = project_model.nv
    # exactly GRID_BATCH DISTINCT states so a per-slot stride bug cannot hide
    samples = _build_cuda_samples(project_model, random_count=_BATCH, include_corner_samples=False)
    samples = samples[:_BATCH]
    assert len(samples) == _BATCH, "need GRID_BATCH distinct samples"

    out = _parse_runner_output(_run_runner(executable, _stdin(samples), cmd))

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected,
                                   rtol=2e-3, atol=max(2e-3, 2e-3 * scale), err_msg=msg)

    for k, sample in enumerate(samples):
        q = np.asarray(sample.q, np.float64)
        tag = f"{robot_id}-{base_mode} slot {k} @ {sample.name}"
        minv_ref = np.asarray(project_model.minv(q), dtype=np.float64)
        close(_normalize_cuda_minv(out[f"MINV{k}"]), minv_ref, f"Minv {tag} (batched host)")
        if f"M{k}" in out:
            m_ref = np.asarray(project_model.crba(q), dtype=np.float64)
            close(_normalize_cuda_minv(out[f"M{k}"]), m_ref, f"M {tag} (batched host)")

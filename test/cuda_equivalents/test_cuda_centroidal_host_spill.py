"""Centroidal HOST-WRAPPER spill CUDA equivalence test (DE-GATE #2 validation).

The standard centroidal equivalence test (test_cuda_centroidal_equivalence.py)
drives the `*_device` functions, which keep the Jw band in smem
(centroidal_inner<T,true>). The com/ccrba/energy KERNELS (host wrappers) instead
carve the Jw band (6*nv*NB) out of s_temp into a SEPARATE tier-routed buffer s_J
(smem tail at the J-in-smem tier, the L2-pinned d_workspace SO band at the
J-spilled tier) and call centroidal_inner<T,false> — a code path NO other test
exercises (dccrba/cmm use their own inners). This test forces the J-spilled rung
by codegen'ing at a low GRID_CUDA_TARGET_SHARED_MEM_BYTES and drives the host
wrappers end-to-end through hd_data->h_* against the RBDReference oracle, so the
<T,false> inner branch + the energy KE reach-back no-J offsets + the d_workspace
repoint are all validated. Restricted codegen (com/ccrba/energy only) keeps the
SO kernels out of the header so the forced-low target does not trigger the
pathological SO deep-spill compile.

Robots: iiwa14-fixed (cheap) + go2-floating (fast-compiling quadruped).
Override with GRID_CUDA_CENTROIDAL_SPILL_ROBOTS="iiwa14:fixed,g1:floating".
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

RUNNER_SOURCE = Path(__file__).with_name("cuda_centroidal_host_spill_runner.cu")
_BATCH = 2
_ALGO_KEYS = ["com", "ccrba", "energy"]
# Force the J-spilled rung at codegen: below any centroidal full arena, so
# select_shared_tier_3way picks the last (Jspill) rung for every tier. Override
# with GRID_CENTROIDAL_FORCE_TARGET (e.g. a high value to exercise the J-in-smem
# tail rung that fitting robots use in production).
_FORCE_TARGET = os.environ.get("GRID_CENTROIDAL_FORCE_TARGET", "2048")
_FORCE_SPILL = int(_FORCE_TARGET) < 8192


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_CENTROIDAL_SPILL_ROBOTS", "iiwa14:fixed,go2:floating")
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
    # The target is read in GRiDCodeGenerator.__init__, so set it BEFORE construction.
    prev = os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES")
    os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = _FORCE_TARGET
    try:
        codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            codegen.gen_all_code(algorithm_list=_ALGO_KEYS, output_path=str(header))
    finally:
        if prev is None:
            os.environ.pop("GRID_CUDA_TARGET_SHARED_MEM_BYTES", None)
        else:
            os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = prev
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_centroidal_host_spill_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-DGRID_BATCH={_BATCH}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA centroidal host-spill runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(samples, nq, nv):
    rows = []
    for s in samples:
        q = np.asarray(s.q, dtype=np.float32).reshape(-1)[:nq]
        qd = np.asarray(s.qd, dtype=np.float32).reshape(-1)[:nv]
        rows.append(" ".join(f"{v:.9g}" for v in np.concatenate([q, qd])))
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_centroidal_host_spill_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_centroidal_spill"
    build_dir.mkdir()
    header = _generate_header(project_model, build_dir)
    # confirm the forced target selected the expected rung (J external vs smem tail).
    htxt = header.read_text()
    expect = "false" if _FORCE_SPILL else "true"
    assert f"COM_J_IN_SMEM() {{ return (TIER == TIER_SHARED) ? {expect}" in htxt, \
        f"forced target did not select the expected com rung (J_IN_SMEM={expect})"
    executable, cmd = _compile_runner(build_dir)

    ref = project_model.reference
    nq, nv = project_model.nq, project_model.nv
    samples = _build_cuda_samples(project_model, random_count=_BATCH, include_corner_samples=False)[:_BATCH]
    assert len(samples) == _BATCH

    out = _parse_runner_output(_run_runner(executable, _stdin(samples, nq, nv), cmd))

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected,
                                   rtol=2e-3, atol=max(2e-3, 2e-3 * scale), err_msg=msg)

    for k, sample in enumerate(samples):
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        m_total, _ = ref._total_mass_and_com(q)
        if not (np.isfinite(m_total) and m_total != 0.0):
            continue
        tag = f"{robot_id}-{base_mode} slot {k} @ {sample.name} (J-spilled host)"

        close(out[f"COM{k}"].reshape(-1), np.asarray(ref.com(q)).reshape(-1), f"{tag} com")
        close(out[f"JCOM{k}"].reshape(3, nv, order="F"),
              np.asarray(ref.jacobian_com(q), dtype=np.float64), f"{tag} jacobian_com")

        A_ref, h_ref = ref.ccrba(q, qd)
        close(out[f"CCRBA_A{k}"].reshape(6, nv, order="F"),
              np.asarray(A_ref, dtype=np.float64), f"{tag} ccrba A")
        close(out[f"CCRBA_H{k}"].reshape(-1),
              np.asarray(h_ref, dtype=np.float64).reshape(-1), f"{tag} ccrba h")

        ke = ref.kinetic_energy(q, qd)
        pe = ref.potential_energy(q)
        me = ref.mechanical_energy(q, qd)
        close(out[f"ENERGY{k}"].reshape(-1), np.array([ke, pe, me]),
              f"{tag} energy [KE, PE, mechanical]")

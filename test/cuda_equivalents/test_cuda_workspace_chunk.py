"""Workspace-chunk bit-identity gate (chunked two-phase batch SO seam).

The GRID_WORKSPACE_CHUNK seam lets the bench's solo SO exes run huge batches
with a chunk-sized workspace arena: chunk-aware host wrappers (idsva_so body /
world, fdsva_so, f_ext_gradient_dq) sweep the batch in C-sized launches,
offsetting input/output pointers per chunk while reusing the arena; outputs stay
full-N on device. Timesteps are independent and no buffer address enters the
arithmetic, so a chunked run must be BIT-identical to an unchunked one — this
test asserts that rather than assuming it.

Both arms compile from the SAME chunk-capable header (emit_workspace_chunking +
emit_alloc_gating, the exact composition per_algo_bench uses) and differ ONLY in
-DGRID_WORKSPACE_CHUNK (C=8 vs 0) at GRID_BATCH=32 — so every wrapper takes 4
full chunks, and the arms disagree loudly if a chunk loop mis-offsets a pointer,
passes the wrong per-launch count, or lets workspace state leak across chunks.
The compile also exercises the GRID_ALLOC_GATE tripwire composition.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

RUNNER_SOURCE = Path(__file__).with_name("cuda_workspace_chunk_runner.cu")
_BATCH = 32
_CHUNK = 8
# canonical gen_all_code keys ("f_ext_gradient" pulls in the _dq surface;
# floating-base pulls in idsva_so_world_frame via its enable default)
_ALGO_KEYS = ["idsva_so_body_frame", "fdsva_so", "f_ext_gradient"]
# alloc-gate unlock keys mirror the init_gridData guards (bench composition)
_GATE_DEFINES = ["GRID_ALLOC_GATE=1", "GRID_ALLOC_IDSVA_SO=1", "GRID_ALLOC_FDSVA_SO=1",
                 "GRID_ALLOC_F_EXT_GRADIENT=1", "GRID_ALLOC_F_EXT_GRADIENT_DQ=1"]


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_WORKSPACE_CHUNK_ROBOTS", "iiwa14:fixed,go2:floating")
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
        codegen.gen_all_code(algorithm_list=_ALGO_KEYS, output_path=str(header),
                             emit_alloc_gating=True, emit_workspace_chunking=True)
    return header


def _compile_runner(build_dir, chunk):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    if not runner_copy.exists():
        shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / f"cuda_workspace_chunk_{chunk}.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-DGRID_BATCH={_BATCH}",
        f"-DGRID_WORKSPACE_CHUNK={chunk}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ] + [f"-D{d}" for d in _GATE_DEFINES]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA workspace-chunk runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


def _run(executable):
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        pytest.fail(f"{executable.name} rc={result.returncode}\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}")
    return result.stdout


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_workspace_chunk_bit_identity(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_workspace_chunk"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)

    # sequential compiles on purpose: SO exes are RAM-heavy to build
    exe_unchunked = _compile_runner(build_dir, 0)
    exe_chunked = _compile_runner(build_dir, _CHUNK)

    out_unchunked = _run(exe_unchunked)
    out_chunked = _run(exe_chunked)

    assert "BEGIN IDSVA_SO" in out_unchunked, "runner produced no output blocks"
    if out_chunked != out_unchunked:
        # locate the first divergent block for a readable failure
        diverged = [name for name in ("IDSVA_SO", "FDSVA_SO", "F_EXT_GRADIENT_DQ")
                    if _block(out_chunked, name) != _block(out_unchunked, name)]
        pytest.fail(
            f"chunked (C={_CHUNK}) vs unchunked outputs differ for {robot_id}-{base_mode}: "
            f"divergent blocks = {diverged or ['<non-block output>']}"
        )


def _block(text, name):
    begin, end = f"BEGIN {name} ", f"END {name}"
    i = text.find(begin)
    j = text.find(end)
    return text[i:j] if i >= 0 and j >= 0 else None

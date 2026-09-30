"""Workspace-slots bit-identity gate (runtime workspace-slot seam).

init_gridData auto-fits gridData.workspace_timestep_slots to remaining device
memory (cudaMemGetInfo; GRID_WORKSPACE_TIMESTEP_SLOTS env override), kernels
index the workspace arena per-BLOCK (grid_workspace_slot()), and every
workspace-using host wrapper clamps its launch grid to the slot count. Timesteps
are independent and per-timestep computation never depends on which block (or
how many blocks) executes it, so a slot-clamped run must be BIT-identical to an
unclamped one — this test asserts that rather than assuming it.

ONE exe (compiled with the bench's alloc-gate composition, multi-block launch
grid) runs three times: env unset (slots == batch, clamp no-op),
GRID_WORKSPACE_TIMESTEP_SLOTS=3 (grid clamped 32 -> 3 blocks; 3 deliberately
does not divide the batch), and =1 (single-slot fully-serialized extreme). The
arms disagree loudly if the per-block slot mapping aliases, the clamp mis-sizes
the grid, or workspace state leaks between a block's grid-stride timesteps.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

RUNNER_SOURCE = Path(__file__).with_name("cuda_workspace_slots_runner.cu")
_BATCH = 32
_FORCED_SLOTS = (3, 1)   # non-divisor clamp + single-slot extreme
# canonical gen_all_code keys ("f_ext_gradient" pulls in the _dq surface;
# floating-base pulls in idsva_so_world_frame via its enable default)
_ALGO_KEYS = ["idsva_so_body_frame", "fdsva_so", "f_ext_gradient"]
# alloc-gate unlock keys mirror the init_gridData guards (bench composition)
_GATE_DEFINES = ["GRID_ALLOC_GATE=1", "GRID_ALLOC_IDSVA_SO=1", "GRID_ALLOC_FDSVA_SO=1",
                 "GRID_ALLOC_F_EXT_GRADIENT=1", "GRID_ALLOC_F_EXT_GRADIENT_DQ=1"]


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_WORKSPACE_SLOTS_ROBOTS", "iiwa14:fixed,go2:floating")
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
                             emit_alloc_gating=True)
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    if not runner_copy.exists():
        shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_workspace_slots.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-DGRID_BATCH={_BATCH}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ] + [f"-D{d}" for d in _GATE_DEFINES]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA workspace-slots runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


def _run(executable, forced_slots=None):
    env = dict(os.environ)
    env.pop("GRID_WORKSPACE_TIMESTEP_SLOTS", None)
    if forced_slots is not None:
        env["GRID_WORKSPACE_TIMESTEP_SLOTS"] = str(forced_slots)
    result = subprocess.run([str(executable)], capture_output=True, text=True,
                            timeout=600, env=env)
    if result.returncode != 0:
        pytest.fail(f"{executable.name} (slots={forced_slots}) rc={result.returncode}\n"
                    f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}")
    # the exe reports the slot count it actually got (stderr, so stdout stays comparable)
    want = _BATCH if forced_slots is None else min(forced_slots, _BATCH)
    assert f"workspace_timestep_slots={want}" in result.stderr, (
        f"exe reported unexpected slot count (wanted {want}):\n{result.stderr[-500:]}")
    return result.stdout


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_workspace_slots_bit_identity(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_workspace_slots"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)

    exe = _compile_runner(build_dir)

    out_unclamped = _run(exe)
    assert "BEGIN IDSVA_SO" in out_unclamped, "runner produced no output blocks"
    for forced in _FORCED_SLOTS:
        out_forced = _run(exe, forced_slots=forced)
        if out_forced != out_unclamped:
            diverged = [name for name in ("IDSVA_SO", "FDSVA_SO", "F_EXT_GRADIENT_DQ")
                        if _block(out_forced, name) != _block(out_unclamped, name)]
            pytest.fail(
                f"slot-clamped (slots={forced}) vs unclamped outputs differ for "
                f"{robot_id}-{base_mode}: divergent blocks = {diverged or ['<non-block output>']}"
            )


def _block(text, name):
    begin, end = f"BEGIN {name} ", f"END {name}"
    i = text.find(begin)
    j = text.find(end)
    return text[i:j] if i >= 0 and j >= 0 else None

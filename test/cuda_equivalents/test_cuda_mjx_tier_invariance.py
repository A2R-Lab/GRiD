"""mjx second-order tier-invariance regression (B3-narrow).

The MuJoCo-convention twins of the second-order kernels (idsva_so, fdsva_so) run
an epilogue that reuses the SO-temp region of d_workspace as `d_mjx_scratch`. At a
SPILLED tier that same region simultaneously backs the pin inner's spilled s_temp
arena (`d_temp_spill`) -- they alias by design (the inner arena is dead once the
epilogue starts; `so_workspace_t_count = max(8*nv^3, idsva_so_spill_ws, ...)` sizes
the region for whichever is live). This test proves that reuse is safe by launching
each mjx kernel at TIER_SHARED (nothing spilled) and TIER_MINIMAL (whole inner arena
spilled, so d_mjx_scratch overlaps the just-freed arena) on IDENTICAL input and
asserting BIT-identical output across a thread sweep (also a thread-invariance
check). A stale read of the spilled arena, or a missing write-before-read barrier
around the reuse, would break bit-identity.

Uses go2-floating (a small floating, non-mimic, non-skew robot -> emits the mjx
twins). mjx is enabled explicitly (the cuda_equivalence conftest defaults pin-only).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_RUNNER = _HERE / "cuda_mjx_tier_invariance_runner.cu"
# idsva_so_world_frame + fdsva_so + the inner deps they recompute in the epilogue
# (id-value, crba, id-gradient) + integrator retraction helpers pulled in by the
# emitted integrator_hessian_device that composes fdsva_so.
_ALGOS = [
    "forward_dynamics", "minv", "inverse_dynamics", "crba",
    "inverse_dynamics_gradient", "forward_dynamics_gradient", "aba",
    "idsva_so_world_frame", "fdsva_so", "integrator", "integrator_gradient",
]
_ARCH = os.environ.get("GRID_CUDA_ARCH", "120")


def _has_cuda() -> bool:
    return shutil.which("nvcc") is not None or Path("/usr/local/cuda/bin/nvcc").exists()


def _gen_header(build_dir: Path) -> None:
    from grid_codegen import GRiDCodeGenerator
    from RBDReference.tests import MANIFEST_PATH
    from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
    from RBDReference.equivalents.reference_backend import build_project_adapter

    spec = [c["spec"] for c in iter_robot_cases(MANIFEST_PATH, base_mode="floating")
            if c["spec"].robot_id == "go2"][0]
    pm = build_project_adapter(spec, resolve_robot_spec(spec), base_mode="floating")
    cg = GRiDCodeGenerator(pm.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        cg.gen_all_code(include_homogenous_transforms=True, output_path=str(build_dir / "grid.cuh"),
                        algorithm_list=_ALGOS, enable_floating_second_order=True,
                        enable_mujoco_kernels=True)


def _compile(build_dir: Path, fdsva: int) -> Path:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    shutil.copyfile(_RUNNER, build_dir / "runner.cu")
    glass = _ROOT / "external" / "GLASS" / "include"
    exe = build_dir / f"runner_{'fdsva' if fdsva else 'idsva'}.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-arch", f"sm_{_ARCH}",
           f"-DMJX_ALGO_FDSVA={fdsva}", "-diag-suppress", "177", "-diag-suppress", "174",
           f"-I{glass}", f"-I{build_dir}", "-o", str(exe), str(build_dir / "runner.cu")]
    res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"mjx tier-invariance runner compile failed (fdsva={fdsva}).\n"
                    f"{' '.join(cmd)}\n{res.stderr[-3000:]}")
    return exe


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.parametrize("algo", ["idsva_so", "fdsva_so"])
def test_mjx_second_order_tier_invariance(tmp_path, algo):
    if not _has_cuda():
        pytest.skip("needs nvcc + CUDA GPU")
    _gen_header(tmp_path)  # one header serves both algos; regen is cheap vs compile
    exe = _compile(tmp_path, fdsva=1 if algo == "fdsva_so" else 0)
    res = subprocess.run([str(exe)], capture_output=True, text=True)
    assert res.returncode == 0 and "RESULT: PASS" in res.stdout, (
        f"{algo} mjx tier-invariance FAILED (rc={res.returncode}):\n{res.stdout}\n{res.stderr[-1500:]}")

"""Library-safe model / joint-limit initialization (HJCD ask 2026-09-21).

Generates a header per robot, compiles ``cuda_safe_init_runner.cu`` against it
TWICE (default fail-fast policy, and -DGRID_GPUERRCHK_NO_EXIT) and drives the
runner's modes. The runner defines the host-only ``GRID_CUDA_CALL`` /
``GRID_HOST_ALLOC`` seams BEFORE including grid.cuh, so every allocation and
copy in the generated ``*_checked`` initializers is fault-injectable
deterministically — no real OOM, no context reset, no runtime cost in the
math kernels (the seams are never used in device code).

Robots: iiwa14 fixed (serial chain, no topology-helper table) and go2
floating with ALL runtime tables on (topology helpers + inertia + transform +
joint-dynamics ownership paths), so the ledger covers every nested member.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

REPO = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path(__file__).with_name("cuda_safe_init_runner.cu")
ASSETS = REPO / "config" / "robot_assets"

CASES = [
    ("iiwa14", False, {}),
    ("go2", True, {"runtime_inertia": True, "runtime_transform": True,
                   "runtime_joint_dynamics": True, "use_joint_dynamics": True}),
]


def _build(name, floating, options, build_dir):
    robot = URDFParser().parse(str(ASSETS / f"{name}.urdf"), floating_base=floating)
    options = dict(options)
    # USE_JOINT_DYNAMICS is a generator-constructor knob (the damping/friction
    # bias); runtime_joint_dynamics (gen_all_code) makes its table mutable.
    use_jd = options.pop("use_joint_dynamics", False)
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid",
                            USE_JOINT_DYNAMICS=use_jd)
    build_dir.mkdir(parents=True, exist_ok=True)
    gen.gen_all_code(algorithm_list=["inverse_dynamics"], output_path=str(build_dir / "grid.cuh"),
                     enable_mujoco_kernels=False, **options)
    header = (build_dir / "grid.cuh").read_text()
    xi = re.search(r"cudaMalloc\(\(void\*\*\)&d_XImats,(\d+)\*sizeof\(T\)\)", header)
    jl = re.search(r"cudaMalloc\(\(void\*\*\)&d_joint_limits,(\d+)\*sizeof\(T\)\)", header)
    assert xi and jl, "could not find the XImats / joint-limit table sizes in grid.cuh"
    # The header must not exit/abort/reset anywhere on the checked path.
    checked = header[header.index("init_robotModel_checked"):header.index("robotModel<T>* init_robotModel()")]
    assert "exit(" not in checked and "abort(" not in checked and "cudaDeviceReset" not in checked
    runner = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner)
    arch = _detect_cuda_arch()
    exes = {}
    for tag, extra in (("default", []), ("noexit", ["-DGRID_GPUERRCHK_NO_EXIT"])):
        exe = build_dir / f"runner_{tag}.exe"
        cmd = ["nvcc", "-std=c++17", "-O1", f"-arch=sm_{arch}", "-I", str(build_dir),
               f"-DGRID_TEST_XI_SIZE={xi.group(1)}", f"-DGRID_TEST_JL_SIZE={jl.group(1)}",
               *extra, "-o", str(exe), str(runner)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"runner compilation ({tag}) failed:\n{r.stderr[-4000:]}"
        exes[tag] = exe
    return exes


def _run(exe, *args):
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize("name,floating,options", CASES, ids=[c[0] for c in CASES])
def test_checked_initialization_contract(tmp_path, name, floating, options):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    exes = _build(name, floating, options, tmp_path / f"{name}_safe_init")
    for mode in ("success", "sweep", "cleanup", "limits"):
        rc, out = _run(exes["default"], mode)
        assert rc == 0 and f"OK {mode}" in out, f"{name}/{mode} (default build):\n{out}"
        rc, out = _run(exes["noexit"], mode)
        assert rc == 0 and f"OK {mode}" in out, f"{name}/{mode} (NO_EXIT build):\n{out}"
    # the sweep must have exercised a real construction sequence
    rc, out = _run(exes["default"], "sweep")
    m = re.search(r"SWEEP calls=(\d+) failed_attempts=(\d+)", out)
    assert m and int(m.group(1)) == int(m.group(2)) >= 4, out


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize("name,floating,options", CASES[:1], ids=[CASES[0][0]])
def test_legacy_policy_is_preserved(tmp_path, name, floating, options):
    """The un-suffixed spellings keep their historical behaviour: fail-fast
    exit in the default build (observed from a subprocess, never inside the
    test runner), sticky-first-error + nullptr under GRID_GPUERRCHK_NO_EXIT."""
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    exes = _build(name, floating, options, tmp_path / f"{name}_legacy")
    rc, out = _run(exes["default"], "legacy", "1")
    assert rc != 0 and "GPUassert" in out and "OK legacy" not in out, out
    rc, out = _run(exes["noexit"], "legacy", "1")
    assert rc == 0 and "LEGACY_NULL" in out and "STICKY 2" in out and "OK legacy" in out, out

"""External-forces (f_ext) 3-way CUDA equivalence: CUDA generated code vs the
pure-Python RBDReference vs pinocchio, for a known nonzero LOCAL-frame f_ext.

This is a dedicated, self-contained companion to
``test_cuda_executable_equivalence.py``. It drives the SAME runner
(``cuda_equivalence_runner.cu``) but with ``GRID_RUNNER_FEXT=1`` so the runner
reads a per-body external force (body-major ``6*NUM_BODIES`` local frame,
[angular; linear]) and emits ``*_fext``-labeled outputs for the dynamics that
thread external forces (rnea / fd / aba / rnea-grad / fd-grad). We feed the
IDENTICAL f_ext to RBDReference and pinocchio and assert all three agree.

Gated iiwa14 (fixed) first, then g1 (floating).
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.tolerances import get_tolerance

from GRiDCodeGenerator import GRiDCodeGenerator

from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")

# (robot_id, base_mode). iiwa14 gated first.
_CASES = [("iiwa14", "fixed"), ("g1", "floating")]


def _build_adapters(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            spec = case["spec"]
            resolved = resolve_robot_spec(spec)
            proj = build_project_adapter(spec, resolved, base_mode=base_mode)
            pin = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
            return spec, proj, pin
    pytest.skip(f"case {robot_id}/{base_mode} not in manifest")


def _gen_and_compile(proj, build_dir, floating_base):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        proj.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    import contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True, output_path=str(header)
        )
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    arch = _detect_cuda_arch()
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    exe = build_dir / "cuda_fext_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"compile failed:\n{' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return exe


def _run(exe, stdin_text):
    result = subprocess.run(
        [str(exe)], input=stdin_text, cwd=exe.parent, capture_output=True, text=True,
        env={**os.environ, "GRID_RUNNER_FEXT": "1"},
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        if "shared-memory request" in combined and "this device supports" in combined:
            pytest.skip("Kernel smem request exceeds this GPU's per-block cap.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_fext_three_way_equivalence(robot_id, base_mode, tmp_path):
    spec, proj, pin = _build_adapters(robot_id, base_mode)
    ref = proj.reference
    nb = ref.robot.get_num_bodies()
    nv = ref.robot.get_num_vel()

    sample = build_dynamics_samples(proj)[1]
    q, qd = sample.q, sample.qd
    u = np.zeros(nv, dtype=np.float64)  # torque input for fd/aba/fd_grad

    # Known nonzero local-frame f_ext, body-major [angular; linear].
    rng = np.random.default_rng(20260530)
    f_ext = [rng.uniform(-3.0, 3.0, size=6) for _ in range(nb)]
    f_ext_flat = np.concatenate(f_ext).astype(np.float64)

    floating = base_mode == "floating"
    exe = _gen_and_compile(proj, tmp_path, floating)

    # stdin matches the runner's read order: q (full project layout, len nq —
    # for floating this is the quaternion form, == CUDA NUM_JOINTS), qd, u, then
    # 6*NUM_BODIES f_ext values (body-major local-frame [angular; linear]).
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float32))
    stdin = "\n".join([row(q), row(qd), row(u), row(f_ext_flat)]) + "\n"

    outputs = _parse_runner_output(_run(exe, stdin))

    # 3-way oracle values (project RBDReference + pinocchio), with f_ext.
    # Memoize the (pure-Python, O(n^2-3)) reference computations so each is run
    # exactly once even though several CUDA outputs share a reference call.
    zeros = np.zeros(nv)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _ref_rnea_grad():
        return proj.rnea_grad(q, qd, zeros, f_ext=f_ext)

    @lru_cache(maxsize=None)
    def _pin_rnea_grad():
        return pin.rnea_grad(q, qd, zeros, f_ext=f_ext)

    @lru_cache(maxsize=None)
    def _ref_fd_grad():
        return proj.forward_dynamics_grad(q, qd, u, f_ext=f_ext)

    def expect(name):
        if name == "inverse_dynamics":
            return (proj.rnea(q, qd, zeros, f_ext=f_ext),
                    pin.rnea(q, qd, zeros, f_ext=f_ext))
        if name == "forward_dynamics":
            return (proj.forward_dynamics(q, qd, u, f_ext=f_ext),
                    pin.aba(q, qd, u, f_ext=f_ext))
        if name == "aba":
            return (proj.aba(q, qd, u, f_ext=f_ext),
                    pin.aba(q, qd, u, f_ext=f_ext))
        if name == "inverse_dynamics_gradient_q":
            return (_ref_rnea_grad()[0], _pin_rnea_grad()[0])
        if name == "inverse_dynamics_gradient_qd":
            return (_ref_rnea_grad()[1], _pin_rnea_grad()[1])
        if name == "forward_dynamics_gradient_q":
            fdg = _ref_fd_grad()[0]
            return (fdg, fdg)  # pinocchio not 3-way checked for fd-grad-with-fext
        if name == "forward_dynamics_gradient_qd":
            fdg = _ref_fd_grad()[1]
            return (fdg, fdg)
        raise KeyError(name)

    # Algorithms whose CUDA fext output we 3-way check. fd/aba grads are checked
    # CUDA-vs-RBDReference (pinocchio has no direct fd-grad-with-fext entry, and
    # the RBDReference fd-grad is itself pinocchio-cross-checked in the Python
    # suite). rnea/aba/fd/rnea-grad are checked against BOTH refs.
    pin_checked = {
        "inverse_dynamics", "forward_dynamics", "aba",
        "inverse_dynamics_gradient_q", "inverse_dynamics_gradient_qd",
    }
    checks = [
        ("inverse_dynamics", "rnea"),
        ("forward_dynamics", "forward_dynamics"),
        ("aba", "aba"),
        ("inverse_dynamics_gradient_q", "rnea_grad"),
        ("inverse_dynamics_gradient_qd", "rnea_grad"),
        ("forward_dynamics_gradient_q", "forward_dynamics"),
        ("forward_dynamics_gradient_qd", "forward_dynamics"),
    ]

    failures = []
    for label, tol_algo in checks:
        cuda_key = label + "_fext"
        assert cuda_key in outputs, f"missing CUDA output {cuda_key}; have {list(outputs)}"
        cuda = np.asarray(outputs[cuda_key], dtype=np.float64).reshape(-1)
        ref_val, pin_val = expect(label)
        ref_val = np.asarray(ref_val, dtype=np.float64).reshape(-1)
        tol = get_tolerance(tol_algo, robot_id=robot_id)
        # float32 CUDA path: scale the absolute tolerance by the value magnitude.
        scale = max(1.0, float(np.max(np.abs(ref_val))) if ref_val.size else 1.0)
        atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # extra float32 headroom
        err_ref = float(np.max(np.abs(cuda - ref_val))) if ref_val.size else 0.0
        if err_ref > atol:
            failures.append(f"{cuda_key}: CUDA-vs-RBDReference maxerr={err_ref:.3e} > {atol:.3e}")
        # also confirm RBDReference == pinocchio (the f_ext convention itself)
        if label in pin_checked:
            pin_val = np.asarray(pin_val, dtype=np.float64).reshape(-1)
            err_pin = float(np.max(np.abs(ref_val - pin_val)))
            ptol = tol.atol + tol.rtol * scale
            if err_pin > ptol:
                failures.append(f"{cuda_key}: RBDReference-vs-pinocchio maxerr={err_pin:.3e} > {ptol:.3e}")

    assert not failures, "f_ext 3-way equivalence failures:\n" + "\n".join(failures)

"""CUDA equivalence test for the FD parameter-gradient emission.

Validates `gen_forward_dynamics_parameter_gradient` (CUDA) against
`RBDReference.forward_dynamics_parameter_gradient` (numpy reference):

    dqdd/dpi = -Minv . Y(q, qd, qdd_actual)   with qdd_actual = FD(q, qd, u)

`pi_i = [m, m*c(3), I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]]` per link (GRiD/URDF basis).
The CUDA kernel composes minv (Minv), the regressor (Y) at the actual
acceleration, and the symmetric-upper -Minv . Y apply.

The runner streams q|qd|u. Since 2026-10-15 the torque is u = ID(q, qd, qdd_sample)
(the integrator suite's torque-driven sampling, docs/agent_debugging_guide.md 7.z31):
the sample's third vector fed in directly as u left every zero-torque sample
(nominal, zero) in free fall on a floating base, where the exact
dqdd/dpi = -Minv Y vanishes (Y is identically zero in free fall), so the comparison
measured float32 forward-dynamics noise amplified by Minv (0.01 to 0.09 on fr3,
rizon4, gen3, fetch, baxter floating; 0.39 on g1) instead of the kernel. With the
torque-driven acceleration every sample has a non-degenerate gradient and the
scale-relative float tolerance applies everywhere; the g1-only atol floor that
covered the degenerate sample is gone.
Output is nv x 10*NUM_BODIES and is NOT a gridData field, so the runner allocates
the output buffer itself. Every manifest robot on both bases except
iiwa14-floating (widened 2026-10-15; see _CASES for the exclusion). Narrow with
GRID_CUDA_FPG_ROBOTS="iiwa14:fixed,g1:floating".
"""

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
    _random_thread_count,
    _run_runner,
    _sample_to_stdin,
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.tolerances import get_tolerance


RUNNER_SOURCE = Path(__file__).with_name("cuda_fd_parameter_gradient_smoke_runner.cu")

# (robot_id, base_mode): every manifest robot on both bases, iiwa14 fixed first.
# g1-floating (nv=35, nb=30 -> ~138 KB at level 0) overflows this GPU's per-block smem
# cap, so it exercises the g1-spill rung: the nv x 10*NB regressor Y spills to the
# L2-pinned d_workspace while Minv + vaf + the inner stay in smem, dropping the arena to
# ~94 KB (under the sm_120 ~99 KB cap). It covers the full 6-DoF free-flyer root path.
# iiwa14-floating is LEFT OUT at the parametrization: the iiwa14 URDF's 'base' link
# has a degenerate/missing <inertial> (zero mass), so its floating-base mass matrix
# is singular and dqdd/dpi = -Minv.Y is not a well-defined oracle (the free-flyer
# root is covered by the other eight floating robots). Narrow with
# GRID_CUDA_FPG_ROBOTS="iiwa14:fixed,g1:floating".
_ALL_ROBOTS = ("iiwa14", "go2", "g1", "h1_2", "fr3", "rizon4", "gen3", "fetch", "baxter")
# h1_2 (both bases) is LEFT OUT at the parametrization (2026-10-16): the parameter
# gradient requests more shared memory than the 101376 bytes this GPU offers per
# block even at the most-spilled tier (181536 bytes floating, measured in the chain
# B sample; the fixed cell trips the same cap), so the cells cannot launch; they
# return through GRID_CUDA_FPG_ROBOTS once a deeper rung fits.
_EXCLUDED = {("iiwa14", "floating"),   # massless root link, see above
             ("h1_2", "fixed"), ("h1_2", "floating")}  # shared-memory cap, see above


def _cases():
    raw = os.environ.get("GRID_CUDA_FPG_ROBOTS")
    if raw:
        out = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            rid, _, mode = tok.partition(":")
            out.append((rid.strip(), (mode.strip() or "fixed")))
        return out
    return [(r, b) for b in ("fixed", "floating") for r in _ALL_ROBOTS if (r, b) not in _EXCLUDED]


_CASES = _cases()


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _generate_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # Lean "fd-param-gradient" profile ({id, minv, fd, regressor,
        # forward_dynamics_parameter_gradient}) instead of "all": emits exactly the kernels this
        # runner needs and skips the heavy second-order kernels (fdsva_so/idsva_so),
        # cutting the floating-base nvcc compile time substantially. Also keeps the
        # header mimic-safe (no refused gradient algos) for consistency.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            codegen_profile="fd-param-gradient",
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_fpg_runner.exe"
    thread_count = _random_thread_count()
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        f"-DGRID_CUDA_FPG_TEST_THREADS={thread_count}",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA FD param-grad runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_fd_parameter_gradient_matches_reference(robot_id, base_mode, tmp_path):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    reference = project_model.reference
    nv = robot.get_num_vel()
    nb = robot.get_num_bodies()

    _generate_header(project_model, tmp_path)
    executable, compile_cmd = _compile_runner(tmp_path, base_mode == "floating")

    samples = _build_cuda_samples(project_model, random_count=2)

    failures = []
    for sample in samples:
        # The runner reads the sample's third vector as the torque u. Drive it with
        # u = ID(q, qd, qdd_sample) so the actual acceleration is the sample's qdd
        # (see the module docstring: u = qdd_sample left the zero-acceleration
        # samples in free fall, where the exact gradient is zero).
        q, qd = sample.q, sample.qd
        u = np.asarray(project_model.inverse_dynamics(q, qd, sample.qdd), dtype=np.float64).reshape(-1)
        sample = dataclasses.replace(sample, qdd=u)
        try:
            stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
        except Exception as exc:  # noqa: BLE001
            combined = str(exc).lower()
            if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
                pytest.skip("CUDA runtime unavailable.")
            raise
        outputs = _parse_runner_output(stdout)
        config = outputs["fpg_config"][0]
        np.testing.assert_allclose(
            config[:3],
            np.asarray([robot.get_num_pos(), nv, nb], dtype=np.float64),
            rtol=0.0, atol=0.0,
            err_msg=f"{robot_id} fpg dimension config @ {sample.name}",
        )

        G_cuda = np.asarray(
            outputs["forward_dynamics_parameter_gradient"], dtype=np.float64
        ).reshape(nv, 10 * nb)

        # numpy reference: dqdd/dpi = -Minv . Y(q,qd,FD(q,qd,u))
        G_ref = np.asarray(
            reference.forward_dynamics_parameter_gradient(q, qd, u, GRAVITY=-9.81), dtype=np.float64
        )
        assert G_cuda.shape == G_ref.shape, (
            f"{robot_id}: CUDA dqdd/dpi shape {G_cuda.shape} != ref {G_ref.shape}"
        )

        tol = get_tolerance("forward_dynamics_parameter_gradient", robot_id=robot_id)
        ref_mag = float(np.max(np.abs(G_ref))) if G_ref.size else 0.0
        scale = max(1.0, ref_mag)
        atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # float32 CUDA headroom

        err = float(np.max(np.abs(G_cuda - G_ref))) if G_ref.size else 0.0
        if err > atol:
            failures.append(
                f"{robot_id} @ {sample.name}: CUDA-vs-reference dqdd/dpi maxerr={err:.3e} > {atol:.3e}"
            )

    assert not failures, "FD param-grad CUDA equivalence failures:\n" + "\n".join(failures)

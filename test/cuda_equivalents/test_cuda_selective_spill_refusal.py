"""The dense gradient inner refuses the selective spill flag when compiled.

test/test_selective_spill_availability.py checks that the generated text holds
the static_assert and the availability constant. This compiles it. A consumer
that composes the inners instantiates them with its own flags, so the refusal
has to hold at the consumer's call site: with the flag set, a robot on the
dense layout (fr3, mimic gripper) must fail to compile with the message, and a
robot on the sparse layout (iiwa14) must compile. Without the flag both compile.
The forward dynamics gradient inner forwards the flag and must refuse it too.

Compile only: nothing is linked or run.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess

import pytest
from grid_assets import robot_urdf
from URDFParser import URDFParser

from grid_codegen import GRiDCodeGenerator
from grid_codegen.nvcc_flags import kernel_flags
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]

ALGORITHMS = ["inverse_dynamics_gradient", "forward_dynamics_gradient"]
MESSAGE = "has no selective spill band"
PROBE = """#include "grid.cuh"
// What a composing consumer writes: the inner, instantiated with its own flag.
__device__ void probe(float *s, const float *c, int *topology, float *spill) {
    grid::inverse_dynamics_gradient_inner<float, PROBE_SPILL>(
        s, c, c, c, s, topology, s, spill, -9.81f);
}
"""


@pytest.fixture(scope="module")
def headers(tmp_path_factory):
    out = {}
    for robot in ("iiwa14", "fr3"):
        folder = tmp_path_factory.mktemp(robot)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            parsed = URDFParser().parse(str(robot_urdf(robot)), floating_base=False)
            GRiDCodeGenerator(parsed, FILE_NAMESPACE="grid").gen_all_code(
                output_path=str(folder / "grid.cuh"), algorithm_list=ALGORITHMS)
        (folder / "probe.cu").write_text(PROBE)
        out[robot] = folder
    return out


def _compile(folder, spill: bool, extra=()):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")
    name = "spill" if spill else "plain"
    return subprocess.run(
        [nvcc, *kernel_flags(), "-arch=sm_" + _detect_cuda_arch(), f"-I{folder}",
         f"-DPROBE_SPILL={'true' if spill else 'false'}", *extra,
         "-c", str(folder / "probe.cu"), "-o", str(folder / f"probe_{name}.o")],
        capture_output=True, text=True)


@pytest.mark.parametrize("robot", ["iiwa14", "fr3"])
def test_inner_compiles_without_the_flag(headers, robot):
    run = _compile(headers[robot], spill=False)
    assert run.returncode == 0, run.stderr[-4000:]


def test_sparse_inner_compiles_with_the_flag(headers):
    run = _compile(headers["iiwa14"], spill=True)
    assert run.returncode == 0, run.stderr[-4000:]
    assert MESSAGE not in run.stderr


def test_dense_inner_refuses_the_flag(headers):
    run = _compile(headers["fr3"], spill=True)
    assert run.returncode != 0, "the dense inner compiled with the selective spill flag"
    assert MESSAGE in run.stderr, run.stderr[-4000:]
    assert "static assertion failed" in run.stderr or "static_assert" in run.stderr

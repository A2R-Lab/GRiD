"""vendor_glass=False (GATO ask 2026-09-20), GLASS provenance label (HJCD
follow-up) and constexpr shared-memory sizers.

* Default generation is byte-identical whether or not the new kwargs are
  passed (the opt-in path changes nothing for existing consumers).
* vendor_glass=False: no vendored GLASS block, a top-level ``#include
  "glass.cuh"`` in the prelude and ``namespace glass = ::glass;`` in the grid
  namespace; the header compiles against the repo's GLASS with
  ``-I external/GLASS`` and the model initializes and runs (the safe-init
  runner's success mode is the TU).
* Provenance: ``glass_revision=`` equal to the checkout's HEAD yields a header
  byte-identical to git discovery; a disagreeing revision is refused.
* ``*_DYNAMIC_SHARED_MEM_BYTES<T>()`` and friends are usable in constant
  expressions (``static_assert``), which is what GATO asked for.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.helpers._lin_alg_helpers import _glass_git_head
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

REPO = Path(__file__).resolve().parents[2]
GLASS_ROOT = REPO / "external" / "GLASS"
RUNNER = Path(__file__).with_name("cuda_safe_init_runner.cu")
URDF = REPO / "config" / "robot_assets" / "iiwa14.urdf"


def _gen(out, **kw):
    robot = URDFParser().parse(str(URDF), floating_base=False)
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    gen.gen_all_code(algorithm_list=["inverse_dynamics", "forward_dynamics"], output_path=str(out),
                     enable_mujoco_kernels=False, **kw)
    return gen, out.read_text()


def test_default_is_byte_identical_with_explicit_defaults(tmp_path):
    _, a = _gen(tmp_path / "a.cuh")
    _, b = _gen(tmp_path / "b.cuh", vendor_glass=True, glass_revision=None)
    assert a == b


def test_supplied_revision_matching_git_is_byte_identical_and_wrong_one_is_refused(tmp_path):
    head = _glass_git_head()
    if head is None:
        pytest.skip("GLASS submodule has no git checkout here")
    gen, a = _gen(tmp_path / "a.cuh")
    assert gen.glass_revision_source == "git"
    gen2, b = _gen(tmp_path / "b.cuh", glass_revision=head)
    assert gen2.glass_revision_source == "git-verified"
    assert a == b
    assert f"// Pinned commit: {head}" in a
    with pytest.raises(ValueError, match="disagrees"):
        _gen(tmp_path / "c.cuh", glass_revision="0" * 40)
    # env var route
    os.environ["GRID_GLASS_REVISION"] = head
    try:
        _, d = _gen(tmp_path / "d.cuh")
    finally:
        del os.environ["GRID_GLASS_REVISION"]
    assert a == d


@pytest.mark.cuda_equivalence
def test_vendor_glass_false_compiles_against_top_level_glass_and_runs(tmp_path):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    build = tmp_path / "alias"; build.mkdir()
    _, h = _gen(build / "grid.cuh", vendor_glass=False)
    assert "// BEGIN GLASS " not in h, "vendored GLASS block emitted under vendor_glass=False"
    assert '#include "glass.cuh"' in h and "namespace glass = ::glass;" in h
    # the alias must sit INSIDE the grid namespace
    assert h.index("namespace grid {") < h.index("namespace glass = ::glass;")
    xi = re.search(r"cudaMalloc\(\(void\*\*\)&d_XImats,(\d+)\*sizeof\(T\)\)", h)
    jl = re.search(r"cudaMalloc\(\(void\*\*\)&d_joint_limits,(\d+)\*sizeof\(T\)\)", h)
    shutil.copyfile(RUNNER, build / RUNNER.name)
    # constexpr sizers: a constant expression must accept them
    (build / "sizers.cu").write_text(
        '#include "grid.cuh"\n'
        "static_assert(grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<float>() > 0, \"sizer is constexpr\");\n"
        "static_assert(grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<float, grid::TIER_SHARED>() > 0, \"tiered sizer is constexpr\");\n"
        "int main(){ return 0; }\n")
    arch = _detect_cuda_arch()
    for src, exe in ((RUNNER.name, "runner.exe"), ("sizers.cu", "sizers.exe")):
        r = subprocess.run(["nvcc", "-std=c++17", "-O1", f"-arch=sm_{arch}", "-I", str(build),
                            "-I", str(GLASS_ROOT), f"-DGRID_TEST_XI_SIZE={xi.group(1)}",
                            f"-DGRID_TEST_JL_SIZE={jl.group(1)}", "-o", str(build / exe), str(build / src)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{src} failed to compile against the top-level GLASS:\n{r.stderr[-4000:]}"
    r = subprocess.run([str(build / "runner.exe"), "success"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK success" in r.stdout, r.stdout + r.stderr

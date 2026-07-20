"""Permanent smem-poison regression (the Inc2 audit, wired as a standing gate).

`cuda_equivalence_runner.cu` has an opt-in `GRID_POISON_SMEM` mode that fills every
SM's dynamic shared memory with NaN (0xFF) BEFORE each algorithm launch. If any
generated device fn reads a caller-carved arena slot before writing it, that NaN
surfaces in the output and fails the golden comparison. `initcheck` is blind to
shared memory and `racecheck` does not flag never-written reads, so this poison
sweep is the only tool that catches the §1j/§1p read-before-write / beta==0-output
class (see docs/agent_debugging_guide.md §1p).

This test re-runs a SMALL slice of the CUDA equivalence suite (one fixed-base + one
floating-base robot, the value-block algo subset, the `zero` sample, one thread count)
with poison ON and asserts it still passes. The value block (inverse/forward dynamics,
minv, crba, aba, end_effector_pose) exercises every caller-carved-arena class while
keeping the header small enough to compile fast — a permanent gate must be cheap. The
FULL sweep (gradients + second-order arenas) is a manual, one-off audit:
`GRID_POISON_SMEM=1 pytest test/cuda_equivalents/test_cuda_executable_equivalence.py`.

We drive it as a nested pytest (rather than duplicating the fixture-heavy
gen/compile/run/oracle plumbing) so there is exactly one source of truth for the
equivalence machinery. The nested run executes in a temp cwd so its gpu-proof.json
receipt is isolated and never clobbers the parent's.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cuda_equivalence

REPO_ROOT = Path(__file__).resolve().parents[2]
EQUIV_TEST = REPO_ROOT / "test/cuda_equivalents/test_cuda_executable_equivalence.py"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available")
@pytest.mark.parametrize("selector", ["iiwa14 and fixed", "go2 and floating"])
def test_equivalence_survives_smem_poison(selector, tmp_path):
    env = dict(os.environ)
    env["GRID_POISON_SMEM"] = "1"
    env["GRID_CUDA_CODEGEN_SUBSET"] = "value"    # value-block header -> fast compile
    env["GRID_CUDA_SAMPLE_NAMES"] = "zero"       # one deterministic sample
    env["GRID_CUDA_RANDOM_SAMPLES"] = "0"
    env["GRID_CUDA_THREAD_COUNTS"] = "32"
    env["PYTHONPATH"] = str(REPO_ROOT)
    # Reuse the parent's compiled-runner cache (poison is runtime-only -> same exe).
    env.setdefault("GRID_CUDA_CACHE_DIR", str(REPO_ROOT / ".pytest_cache/grid_cuda"))

    cmd = [
        sys.executable, "-m", "pytest", str(EQUIV_TEST),
        "-k", selector, "-q",
        "-p", "no:cacheprovider",
    ]
    # cwd=tmp_path isolates the nested run's gpu-proof.json receipt from the repo.
    result = subprocess.run(cmd, env=env, cwd=str(tmp_path), capture_output=True, text=True)
    tail = f"STDOUT:\n{result.stdout[-4000:]}\n\nSTDERR:\n{result.stderr[-2000:]}"

    # rc 5 = "no tests collected" (selector matched nothing on this manifest) -> skip.
    if result.returncode == 5:
        pytest.skip(f"no equivalence case matched {selector!r}\n{tail}")
    # A GPU-unavailable environment surfaces as a skip inside the nested run.
    if result.returncode != 0 and "cuda runtime is unavailable" in tail.lower():
        pytest.skip("CUDA runtime unavailable in nested equivalence run")

    assert result.returncode == 0, (
        f"CUDA equivalence FAILED under GRID_POISON_SMEM=1 for {selector!r} "
        f"-> a device fn reads caller-carved smem before writing it "
        f"(read-before-write / beta==0-output; see docs §1p).\n{tail}"
    )

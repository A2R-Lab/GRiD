"""CUDA-equivalence suite conftest: generate PIN-ONLY headers by default.

WHY (2026-07-24). Every test in this directory generates a `grid.cuh` and compiles it
with nvcc. On a floating-base, non-mimic robot the generator also instantiates the mjx
(`MUJOCO_OUTPUT=true`) twin of each kernel -- and those twins are enormous next to their
pin counterparts. Measured on g1-floating:

    idsva_so_world_frame      28.1x        forward/inverse dynamics, minv, crba ~1.0x
    fdsva_so                   4.5x
    inverse_dynamics_gradient  2.9x

~2.1M of the ~2.4M SASS lines in a humanoid build are mjx-only. That cost was being paid
by this suite on every floating cell.

It bought nothing: **no test in this directory exercises mjx.** (Audited 2026-07-24 --
the only `mujoco`/`MUJOCO` hits under `test/cuda_equivalents/` are four comments in .cu
runners; not one runner launches an mjx kernel and not one .py test asserts against the
MuJoCo output convention.) The suite was compiling the expensive half of the library and
then testing the other half. Same worst-of-both-worlds shape as the benchmark harness,
which compiled every mjx twin and never timed one.

So: default this suite to pin-only. Correctness coverage is unchanged, because there was
no mjx coverage to lose.

OPTING BACK IN. Wave B adds real mjx coverage. A test that genuinely exercises mjx must
pass `enable_mujoco_kernels=True` EXPLICITLY to its own `gen_all_code` call -- an explicit
argument always beats the env var, so such a test is self-contained and unaffected by this
file. Do not rely on the env default for mjx coverage; a test that silently generated a
pin-only header would pass vacuously.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _pin_only_headers():
    """Default `gen_all_code` to `enable_mujoco_kernels=False` for this directory.

    Honors a caller-set `GRID_ENABLE_MUJOCO_KERNELS` (e.g. a deliberate
    `GRID_ENABLE_MUJOCO_KERNELS=1` sweep) rather than overriding it.
    """
    key = "GRID_ENABLE_MUJOCO_KERNELS"
    preset = os.environ.get(key)
    if preset is None:
        os.environ[key] = "0"
    try:
        yield
    finally:
        if preset is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = preset

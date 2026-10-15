"""The capable-box suite floor (test/conftest.py::_ENV_GUARD_SKIP) must match the
skip strings the harness ACTUALLY emits — the 2026-10-13 audit found it matched
"nvcc not found" but not "nvcc was not found", so the floor was inert."""
import re
import pytest

from test.conftest import _ENV_GUARD_SKIP

REAL_SKIP_STRINGS = [
    "nvcc was not found",                       # cuda_harness.py / executable_cache.py
    "nvcc not found on PATH",
    "nvcc is unavailable", "nvcc not available",  # test_cuda_momentum_contract / test_smem_poison
    "CUDA runtime is unavailable",               # cuda_harness.py
    "iiwa14 was not found in the robot manifest",
    "go2 not found in robot manifest",
    "fr3 not in the robot manifest",
    "iiwa14 URDF not present",
    "URDF parse failed",
]
HARDWARE_LIMIT_STRINGS = [   # REAL skips on a capable box; must NOT trip the floor
    "shared-memory request 120 KB > cap 99 KB",
    "too many resources requested for launch",
]


@pytest.mark.parametrize("text", REAL_SKIP_STRINGS)
def test_floor_matches_real_environment_skips(text):
    assert _ENV_GUARD_SKIP.search(text), text


@pytest.mark.parametrize("text", HARDWARE_LIMIT_STRINGS)
def test_floor_ignores_hardware_limit_skips(text):
    assert not _ENV_GUARD_SKIP.search(text), text

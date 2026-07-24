"""Guards the pin-only default that `conftest.py` establishes for this suite.

`conftest.py` sets `GRID_ENABLE_MUJOCO_KERNELS=0` so every header generated here is
PIN-ONLY. That is worth ~2.1M of ~2.4M SASS lines on a floating-base humanoid cell, and
it costs no coverage (no test in this directory exercises mjx -- see conftest.py).

The failure mode this catches is silent: delete or break the conftest fixture and every
test here still PASSES, just far slower, while compiling a mjx half nothing asserts
against. Nothing would go red. Hence an explicit assertion on the contract.

Compile-free by construction (no nvcc, no codegen), so it is allowlisted in
`test/test_marker_hygiene.py` and runs even under `-m "not cuda_equivalence"` -- which is
exactly when you want to know the suite's default is intact.
"""

import os


def test_suite_defaults_to_pin_only_headers():
    value = os.environ.get("GRID_ENABLE_MUJOCO_KERNELS")
    assert value == "0", (
        "The CUDA-equivalence suite must generate PIN-ONLY headers, but "
        f"GRID_ENABLE_MUJOCO_KERNELS is {value!r} (expected '0'). The session fixture in "
        "test/cuda_equivalents/conftest.py sets it; if that file was removed or its "
        "autouse fixture stopped applying, every floating-base cell here silently went "
        "back to compiling the mjx kernel twins -- ~28x the SASS on idsva_so_world_frame "
        "alone -- for coverage no test in this directory asserts against."
    )

"""Regression sentinel for the shared `gen_matmul` block-wrap modulus.

BUG (fixed 2026-05-31): the shared matmul helper emitted by
`grid_codegen/helpers/_lin_alg_helpers.py:gen_matmul` computed the
per-block base offset as

    int cur = 36*((index/num)%NUM_JOINTS);   # <-- WRONG for mimic robots

For MIMIC robots NUM_BODIES > NUM_JOINTS (mimic joints carry 0 DoF but still
add a body), and `matmul` is called over 36*NUM_BODIES elements by the
idsva_so composite-inertia build (I @ Xup). With the `%NUM_JOINTS` modulus the
LAST mimic body wrapped back to block 0 and multiplied against body-0's
inertia, corrupting that body's composite inertia and propagating the error up
the entire composite-inertia chain -> wrong second-order (idsva_so / fdsva_so)
output. This was caught only by the fr3 second-order comparison.

The fix uses `%NUM_BODIES`. NUM_BODIES == NUM_JOINTS for every non-mimic
fixed-base robot, so the change is byte-identical there; only mimic robots
(fr3, h1_2) are affected.

This is a fast codegen-string sentinel (no nvcc): it would FAIL if the modulus
regressed to `%NUM_JOINTS`. It is scoped to fr3-fixed, where NUM_BODIES (9) and
NUM_JOINTS (8) DIFFER, so the assertion genuinely distinguishes the two forms.
It does NOT duplicate the numeric fr3 second-order equivalence coverage in
`test_cuda_executable_equivalence.py` (which needs nvcc + a GPU and only runs
in the developer CUDA sweep) -- it is the cheap always-runnable guard for THIS
specific layout bug.
"""

import contextlib
import os
import re

import pytest

from grid_codegen import GRiDCodeGenerator
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


# The matmul block-wrap is exercised by the idsva_so composite-inertia build, so
# the second-order algorithm must be in the codegen list for the helper to be
# meaningful (the helper is emitted regardless, but listing it keeps this test
# honest about WHY the helper exists).
_MIMIC_SO_ALGORITHM_LIST = ["inverse_dynamics", "crba", "idsva_so_body_frame", "fdsva_so"]

_MATMUL_DEF_RE = re.compile(
    r"void matmul\(int index[^\n]*\)\s*\{(?P<body>.*?)\n\s*\}", re.S
)
_CONST_RE = re.compile(r"const int (?P<name>[A-Z0-9_]+) = (?P<value>-?[0-9]+);")


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} was not found in the robot manifest.")


def _generate_fr3_fixed_header(tmp_path):
    spec = _robot_spec("fr3", "fixed")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            "Could not resolve manifest fr3 (mimic robot). Run "
            f"./install/developer_install.sh before this test. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="fixed")
    header_path = tmp_path / "fr3_fixed_matmul_blockwrap.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=False,
        FILE_NAMESPACE="grid",
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            codegen_profile="all",
            algorithm_list=_MIMIC_SO_ALGORITHM_LIST,
            output_path=str(header_path),
        )
    return header_path.read_text()


def _constants(header):
    return {m.group("name"): int(m.group("value")) for m in _CONST_RE.finditer(header)}


def test_fr3_is_a_meaningful_mimic_sentinel(tmp_path):
    """Precondition: fr3-fixed must have NUM_BODIES > NUM_JOINTS, otherwise the
    `%NUM_BODIES` vs `%NUM_JOINTS` distinction is vacuous and this file is not a
    real regression guard."""
    header = _generate_fr3_fixed_header(tmp_path)
    consts = _constants(header)
    assert "NUM_JOINTS" in consts and "NUM_BODIES" in consts
    assert consts["NUM_BODIES"] > consts["NUM_JOINTS"], (
        "fr3-fixed is expected to be a mimic robot with NUM_BODIES > NUM_JOINTS "
        f"(got NUM_BODIES={consts.get('NUM_BODIES')}, "
        f"NUM_JOINTS={consts.get('NUM_JOINTS')}); without that inequality this "
        "sentinel cannot distinguish the buggy %NUM_JOINTS wrap from the fix."
    )


def test_matmul_block_wrap_uses_num_bodies_not_num_joints(tmp_path):
    """The shared matmul helper must wrap the per-block base offset by
    NUM_BODIES. A regression to %NUM_JOINTS silently corrupts the mimic
    composite-inertia chain (fr3/h1_2 second-order) and MUST fail here."""
    header = _generate_fr3_fixed_header(tmp_path)
    match = _MATMUL_DEF_RE.search(header)
    assert match is not None, (
        "Could not locate the generated `void matmul(int index, ...)` helper in "
        "the fr3-fixed header; the codegen layout changed -- update this sentinel."
    )
    body = match.group("body")
    assert "%NUM_JOINTS" not in body, (
        "REGRESSION: the matmul block-wrap modulus is `%NUM_JOINTS`. For mimic "
        "robots NUM_BODIES > NUM_JOINTS, so the last mimic body wraps to block 0 "
        "and reads body-0's inertia, corrupting the composite-inertia chain and "
        "the second-order (idsva_so/fdsva_so) output. It must be `%NUM_BODIES`.\n"
        f"matmul body was:\n{body}"
    )
    assert "%NUM_BODIES" in body, (
        "The matmul block-wrap modulus must be `%NUM_BODIES` (the per-block base "
        f"offset `36*((index/num)%NUM_BODIES)`). matmul body was:\n{body}"
    )

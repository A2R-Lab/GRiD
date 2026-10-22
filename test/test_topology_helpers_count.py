"""Topology helpers: the count rule, the T-element footprint and the debug assert.

`TOPOLOGY_HELPERS_COUNT` is `(not serial) * (5n + 1) + (not identical S) * n`
with `n = max(NUM_POS, NUM_JOINTS)`; `TOPOLOGY_HELPERS_T_COUNT<T>()` is the
block's footprint in T elements, exact for 4-byte T and a ceiling for 8-byte T
against the arena byte sizer; a DEBUG_MODE header asserts the helper pointer is
not null at the top of every helper-reading `_inner`, and a release header
carries no such assert. Checked on iiwa14 (0), rizon4 (n = 7) and go2 floating
(branching).
"""
from __future__ import annotations

import contextlib
import os
import re

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_assets import robot_urdf
from URDFParser import URDFParser

_COUNT_RE = re.compile(r"const int TOPOLOGY_HELPERS_COUNT = (\d+);")
_TCOUNT_RE = re.compile(r"constexpr size_t TOPOLOGY_HELPERS_T_COUNT\(\) \{ return \(sizeof\(int\) \* static_cast<size_t>\(TOPOLOGY_HELPERS_COUNT\) \+ sizeof\(T\) - 1\) / sizeof\(T\); \}")
_ASSERT = "assert(s_topology_helpers != nullptr);"
_ALGOS = ["inverse_dynamics", "inverse_dynamics_gradient"]


def _gen(tmp_path, name, floating, debug):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
        robot = URDFParser().parse(str(robot_urdf(name)), floating_base=floating)
        gen = GRiDCodeGenerator(robot, debug, False, "grid")
        out = tmp_path / f"{name}_{'debug' if debug else 'release'}.cuh"
        gen.gen_all_code(output_path=str(out), algorithm_list=_ALGOS, enable_mujoco_kernels=False)
    return gen, out.read_text()


def _expected_count(robot):
    n = max(robot.get_num_pos(), robot.get_num_joints())
    count = 0 if robot.is_serial_chain() else 5 * n + 1
    if not robot.are_Ss_identical(list(range(robot.get_num_pos()))):
        count += n
    return count


def _arena_bytes(t_count, int_count, size_t):
    """grid_shared_arena_bytes<T>(t_count, int_count)."""
    def align(x, a):
        return (x + a - 1) // a * a
    offset = align(0, size_t) + size_t * t_count
    if int_count > 0:
        offset = align(offset, 4) + 4 * int_count
    return align(offset, 16)


@pytest.mark.parametrize("name,floating,expected", [("iiwa14", False, 0), ("rizon4", False, 7), ("go2", True, None)],
                         ids=["iiwa14", "rizon4", "go2"])
def test_count_rule_and_t_count(tmp_path, name, floating, expected):
    gen, text = _gen(tmp_path, name, floating, debug=False)
    count = int(_COUNT_RE.search(text).group(1))
    assert count == _expected_count(gen.robot)
    if expected is not None:
        assert count == expected
    else:
        assert not gen.robot.is_serial_chain() and count >= 5 * gen.robot.get_num_joints() + 1
    assert _TCOUNT_RE.search(text), "TOPOLOGY_HELPERS_T_COUNT<T>() missing next to the count"
    for size_t in (4, 8):
        t_count_helpers = (4 * count + size_t - 1) // size_t
        for t in (0, 1, 7, 100, 701):
            exact = _arena_bytes(t, count, size_t)
            folded = _arena_bytes(t + t_count_helpers, 0, size_t)
            if size_t == 4:
                assert folded == exact, (name, t)
            else:
                assert folded >= exact and folded - exact <= 16, (name, t)
    assert _ASSERT not in text, "release header must carry no helper assert"


def test_debug_header_asserts_only_when_helpers_exist(tmp_path):
    _, rizon = _gen(tmp_path, "rizon4", False, debug=True)
    assert rizon.count(_ASSERT) >= 2
    # the assert is the first body line of a helper-reading inner
    m = re.search(r"void inverse_dynamics_inner\([^\n]*\{\n\s*" + re.escape(_ASSERT), rizon)
    assert m, "inverse_dynamics_inner lacks the assert as its first line"
    _, iiwa = _gen(tmp_path, "iiwa14", False, debug=True)
    assert _ASSERT not in iiwa

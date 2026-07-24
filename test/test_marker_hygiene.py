"""Regression sentinel for pytest MARKER placement.

BUG (fixed 2026-07-24): `test/cuda_equivalents/test_cuda_second_order_fallback.py`
carried

    @pytest.mark.cuda_equivalence
    @pytest.mark.developer_only
    @pytest.mark.robot_smoke
    def _flatten_second_order_tensors(tensors):   # <-- a HELPER, not a test

A helper had been inserted between the decorators and the test they were meant to
decorate. pytest silently ignores markers on non-test functions, so the markers were
DEAD and the module's two real tests were left completely unmarked. Both of them
shell out to nvcc and parametrize over g1/h1_2 fixed AND floating -- the heaviest
compiles in the repo. Consequence: `-m "not cuda_equivalence"`, the filter everyone
reaches for to get a fast CPU-only run, still collected 10 nvcc cells and ran for
HOURS (observed: 22 tests in 27 min, a single nvcc alive 661 s).

Nothing failed. The suite was green the whole time -- it was just unusable, which is
why this went unnoticed. Hence a sentinel rather than a one-line fix.

Two checks:

1. **Markers must land on tests** (AST, tree-wide, milliseconds). A `pytest.mark.*`
   decorator on a function that is neither a test nor a fixture is always a mistake --
   pytest ignores it. This is the generalized, syntactic form of the bug above.

2. **The CPU-only filter must be compile-free** (real pytest collection, ~2 s). Asserts
   that `-m "not cuda_equivalence"` collects nothing from `test/cuda_equivalents/`
   except the allowlisted compile-free sentinels. This deliberately uses a real
   collection rather than AST: the marker is applied four different ways across the
   suite -- a plain decorator, a module-level `pytestmark` scalar, a module-level
   `pytestmark` list, and a dynamic `marks.append(...)` inside `parametrize` -- and only
   pytest itself resolves all four. An AST version of this check reported every one of
   the latter three as unmarked.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

_TEST_ROOT = pathlib.Path(__file__).resolve().parent

# Modules under test/cuda_equivalents/ that are deliberately NOT cuda_equivalence:
# pure codegen-string sentinels that never invoke nvcc, kept next to the CUDA tests
# because that is the surface they guard. Each must stay genuinely compile-free.
_COMPILE_FREE_IN_CUDA_EQUIVALENTS = {
    "test_cuda_matmul_blockwrap_regression.py",
}

# Decorators that are plumbing, not markers -- legal on helpers and fixtures.
_NON_MARKER_ATTRS = {"parametrize", "fixture", "usefixtures"}


def _iter_test_modules():
    for path in sorted(_TEST_ROOT.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _marker_names(node):
    """pytest.mark.<name> decorators on `node`, ignoring plumbing decorators."""
    names = []
    for dec in node.decorator_list:
        # Unwrap a call, e.g. @pytest.mark.skipif(...) -> the attribute chain.
        target = dec.func if isinstance(dec, ast.Call) else dec
        if not isinstance(target, ast.Attribute):
            continue
        name = target.attr
        parent = target.value
        if isinstance(parent, ast.Attribute) and parent.attr == "mark":
            if name not in _NON_MARKER_ATTRS:
                names.append(name)
    return names


def _is_fixture(node):
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        attr = getattr(target, "attr", None)
        if attr in {"fixture", "yield_fixture"}:
            return True
    return False


def _functions(path):
    """Top-level and class-level function defs, with their names."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


@pytest.mark.parametrize(
    "module_path",
    list(_iter_test_modules()),
    ids=lambda p: str(p.relative_to(_TEST_ROOT)),
)
def test_markers_are_attached_to_tests_not_helpers(module_path):
    """A pytest.mark.* on a non-test, non-fixture function is silently ignored."""
    offenders = []
    for node in _functions(module_path):
        if node.name.startswith("test_") or _is_fixture(node):
            continue
        markers = _marker_names(node)
        if markers:
            offenders.append(f"{node.name} (line {node.lineno}) carries {markers}")
    assert not offenders, (
        f"{module_path.relative_to(_TEST_ROOT)} applies pytest markers to functions that "
        "are neither tests nor fixtures. pytest IGNORES these, so the markers are dead and "
        "the tests they were meant for are left unmarked (this is exactly how the "
        "second_order_fallback nvcc cells escaped every marker filter). Move the markers "
        "onto the test function:\n  " + "\n  ".join(offenders)
    )


def test_cpu_only_filter_collects_no_cuda_equivalence_tests():
    """`-m "not cuda_equivalence"` must not collect anything that invokes nvcc.

    Collection only -- nothing is executed, so this never compiles.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(_TEST_ROOT / "cuda_equivalents"),
            "--collect-only", "-q", "-p", "no:cacheprovider",
            "-m", "not cuda_equivalence",
        ],
        capture_output=True,
        text=True,
        cwd=_TEST_ROOT.parent,
        timeout=600,
    )
    # rc 5 == "no tests collected", the ideal outcome once everything is marked.
    assert proc.returncode in (0, 5), (
        f"collection failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    survivors = [line for line in proc.stdout.splitlines() if "::" in line]
    leaked = [
        s for s in survivors
        if pathlib.Path(s.split("::")[0]).name not in _COMPILE_FREE_IN_CUDA_EQUIVALENTS
    ]
    assert not leaked, (
        'A "CPU-only" run (-m "not cuda_equivalence") still collects '
        f"{len(leaked)} test(s) from cuda_equivalents/, which will invoke nvcc and can "
        "run for HOURS on the big floating robots. Mark them cuda_equivalence, or -- if "
        "genuinely compile-free -- add the module to _COMPILE_FREE_IN_CUDA_EQUIVALENTS "
        f"in {pathlib.Path(__file__).name}:\n  " + "\n  ".join(leaked)
    )

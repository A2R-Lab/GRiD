"""One nvcc flag list (grid_codegen/nvcc_flags.py) for the wrapper, the native
tests and the examples.

The flags -ftz, -prec-div, -prec-sqrt and -fmad change device-code floating
point. When each call site wrote its own list, the native tests compiled their
kernels differently from the robot libraries the Python wrapper builds. This
module fails when a scanned file states an optimisation or floating-point flag
itself, and pins the shared list, its override and the cache keys that must
follow the override. CPU only: nothing here runs nvcc.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from grid_codegen import nvcc_flags

REPO = Path(__file__).resolve().parents[1]

# Trees whose nvcc commands must come from the shared module.
SCANNED_ROOTS = (
    "test/cuda_equivalents",      # the native tests, cuda_harness.py and executable_cache.py
    "test/python_wrappers",
    "test/diagnostics",
    "bindings/grid_rbd",
    "examples",
)
# Explicitly outside the rule:
#   test/benchmarks            timing code keeps the flags its published numbers were
#                              measured with; it also compiles host-only C++ (g++) whose
#                              -O level is unrelated to the kernel list. Not scanned.
#   test/diagnostics/archive   frozen one-off studies, kept as they were run.
ALLOWED_PREFIXES = (
    "test/benchmarks/",
    "test/diagnostics/archive/",
)
_FLAG = re.compile(r"^(?:-O(?:[0-3sg]|fast)?|--?(?:ftz|prec-div|prec-sqrt|fmad)(?:=.*)?)$")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _flag_literals(source: str) -> list[tuple[int, str]]:
    """(line, token) for every string literal token that is an nvcc optimisation or
    floating-point flag. Comments and docstrings are prose and are not literals."""
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            hits.extend((node.lineno, tok) for tok in node.value.split() if _FLAG.match(tok))
    return hits


def _scanned_sources():
    for root in SCANNED_ROOTS:
        for path in sorted((REPO / root).rglob("*")):
            rel = path.relative_to(REPO).as_posix()
            if rel.startswith(ALLOWED_PREFIXES) or "__pycache__" in path.parts or not path.is_file():
                continue
            if path.suffix == ".py":
                yield rel, path.read_text()
            elif path.suffix == ".ipynb":
                cells = json.loads(path.read_text()).get("cells", [])
                code = ["".join(c["source"]) for c in cells if c.get("cell_type") == "code"]
                # Notebook magics are not Python; no tracked notebook uses one in a compile cell.
                yield rel, "\n".join(src for src in code if not re.search(r"^\s*[%!]", src, re.M))


def test_scanner_sees_flags_and_ignores_prose():
    src = '"""doc -O0"""\n# comment -O2\ncmd = ["nvcc", "-O2", f"-ftz={x}", "a -prec-div=false b", "-fmad=true", "-Omit"]\n'
    assert sorted(tok for _, tok in _flag_literals(src)) == ["-O2", "-fmad=true", "-ftz=", "-prec-div=false"]


def test_no_hard_coded_nvcc_flags_outside_the_shared_module():
    offenders = []
    count = 0
    for rel, source in _scanned_sources():
        count += 1
        offenders.extend(f"{rel}:{line}: {tok}" for line, tok in _flag_literals(source))
    assert count > 100, "the scan found too few files; SCANNED_ROOTS no longer matches the tree"
    assert not offenders, (
        "nvcc optimisation or floating-point flags written at a call site. Build the command from "
        "grid_codegen.nvcc_flags.kernel_flags() instead:\n  " + "\n  ".join(offenders))


def test_example_shell_scripts_take_the_shared_list():
    for path in sorted((REPO / "examples").rglob("*.sh")):
        text = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
        if "NVCC" not in text and "nvcc" not in text:
            continue
        rel = path.relative_to(REPO).as_posix()
        assert "grid_codegen.nvcc_flags" in text, f"{rel} compiles with nvcc without the shared flag list"
        bad = [tok for tok in text.split() if _FLAG.match(tok) or tok.startswith("-std=")]
        assert not bad, f"{rel} states {bad} itself"


def test_default_list_is_the_shipped_one():
    expected = ["-std=c++17", "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-fmad=true"]
    assert list(nvcc_flags.KERNEL_FLAGS) == expected
    assert nvcc_flags.kernel_flags({}) == expected
    assert nvcc_flags.kernel_flags({nvcc_flags.FP_FLAGS_ENV: " default "}) == expected
    assert nvcc_flags.kernel_flags({}) is not nvcc_flags.kernel_flags({})


def test_override_replaces_only_the_floating_point_flags():
    env = nvcc_flags.FP_FLAGS_ENV
    assert env == "GRID_NVCC_FP_FLAGS"
    head = ["-std=c++17", "-O3"]
    assert nvcc_flags.kernel_flags({env: "ieee"}) == head + [
        "-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=true"]
    assert nvcc_flags.kernel_flags({env: "-fmad=false -ftz=false"}) == head + [
        "-ftz=false", "-prec-div=false", "-prec-sqrt=false", "-fmad=false"]
    for bad in ("-O0", "-ftz", "-ftz=maybe", "-ftz=true -ftz=false", "fast"):
        with pytest.raises(ValueError, match=env):
            nvcc_flags.kernel_flags({env: bad})


def test_module_prints_the_list(capsys):
    assert nvcc_flags.main([]) == 0
    assert capsys.readouterr().out.split() == nvcc_flags.kernel_flags()


def test_wrapper_flags_are_the_shared_list_plus_fpic(monkeypatch):
    from grid_rbd import _compile
    monkeypatch.delenv(nvcc_flags.FP_FLAGS_ENV, raising=False)
    assert _compile._nvcc_default_flags() == [*nvcc_flags.KERNEL_FLAGS, "--compiler-options=-fPIC"]
    monkeypatch.setenv(nvcc_flags.FP_FLAGS_ENV, "ieee")
    assert _compile._nvcc_default_flags() == [
        "-std=c++17", "-O3", *nvcc_flags.IEEE_FP_FLAGS, "--compiler-options=-fPIC"]


def test_wrapper_store_keys_follow_the_override(monkeypatch, tmp_path):
    """Stage-1 identity and stage-2 content key: neither hashes the compile command,
    so both fold the resolved list explicitly."""
    from grid_rbd import _cache
    for name in ("grid.cuh", "wrapper.cu", "wrapper_common.cuh"):
        (tmp_path / name).write_text("// " + name + "\n")
    monkeypatch.delenv(nvcc_flags.FP_FLAGS_ENV, raising=False)
    identity = _cache.build_identity(120)
    content = _cache.compute_content_key(tmp_path, {}, 120, 16)
    assert identity["nvcc_kernel_flags"] == list(nvcc_flags.KERNEL_FLAGS)
    monkeypatch.setenv(nvcc_flags.FP_FLAGS_ENV, "ieee")
    assert _cache.build_identity(120)["nvcc_kernel_flags"] != identity["nvcc_kernel_flags"]
    assert _cache.compute_content_key(tmp_path, {}, 120, 16) != content


def test_wrapper_unit_object_key_follows_the_override(monkeypatch, tmp_path):
    from grid_rbd import _compile
    for name in ("grid.cuh", "wrapper_common.cuh", "wrapper_core.cu"):
        (tmp_path / name).write_text("// " + name + "\n")
    unit = tmp_path / "wrapper_core.cu"
    monkeypatch.delenv(nvcc_flags.FP_FLAGS_ENV, raising=False)
    default = _compile._unit_object_key(unit, tmp_path, ["nvcc", *_compile._nvcc_default_flags()])
    monkeypatch.setenv(nvcc_flags.FP_FLAGS_ENV, "ieee")
    assert _compile._unit_object_key(unit, tmp_path, ["nvcc", *_compile._nvcc_default_flags()]) != default


def test_native_executable_key_follows_the_override(tmp_path):
    from test.cuda_equivalents.executable_cache import executable_cache_key
    src = tmp_path / "runner.cu"
    src.write_text("int main() { return 0; }\n")
    env = nvcc_flags.FP_FLAGS_ENV
    default, _ = executable_cache_key([src], nvcc_flags.kernel_flags({}), nvcc="/fake/nvcc")
    ieee, _ = executable_cache_key([src], nvcc_flags.kernel_flags({env: "ieee"}), nvcc="/fake/nvcc")
    assert default != ieee

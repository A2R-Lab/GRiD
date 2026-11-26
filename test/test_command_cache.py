"""CPU-only tests for the content cache around the native tests' own nvcc commands.

A fake nvcc on PATH stands in for the toolkit: it writes an output derived from
its source bytes and logs every invocation, so hits, rebuilds and failures are
observable without a GPU or a real compile.
"""
from __future__ import annotations

import os
import stat
import threading

import pytest

from test.cuda_equivalents.command_cache import cached_nvcc_run, command_cache_key

_FAKE_NVCC = """#!/usr/bin/env bash
if [ "$1" = "--version" ]; then echo "${FAKE_NVCC_VERSION:-fake nvcc 1.0}"; exit 0; fi
echo "$*" >> "$FAKE_NVCC_LOG"
[ -n "$FAKE_NVCC_SLEEP" ] && sleep "$FAKE_NVCC_SLEEP"
[ -n "$FAKE_NVCC_FAIL" ] && { echo "boom" >&2; exit 1; }
echo "warning: unused variable" >&2
out=""; prev=""; src=""
for a in "$@"; do
  [ "$prev" = "-o" ] && out="$a"
  case "$a" in *.cu) src="$a";; esac
  prev="$a"
done
{ sha256sum < "$src"; echo "$FAKE_NVCC_STAMP"; } > "$out"
chmod +x "$out"
"""


@pytest.fixture
def kit(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvcc = bin_dir / "nvcc"
    nvcc.write_text(_FAKE_NVCC)
    nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "nvcc.log"
    log.write_text("")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NVCC_LOG", str(log))
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GRID_CUDA_DISABLE_CACHE", raising=False)
    for knob in ("FAKE_NVCC_FAIL", "FAKE_NVCC_SLEEP", "FAKE_NVCC_VERSION", "FAKE_NVCC_STAMP"):
        monkeypatch.delenv(knob, raising=False)
    shared = tmp_path / "shared"
    (shared / "glass").mkdir(parents=True)
    (shared / "glass" / "ops.h").write_text("// ops v1\n")

    class Kit:
        root = tmp_path
        library = shared / "glass" / "ops.h"

        def tree(self, name):
            """One test's private build directory, the way the modules lay it out."""
            folder = tmp_path / name
            folder.mkdir(exist_ok=True)
            (folder / "runner.cu").write_text('#include "grid.cuh"\nint main() { return 0; }\n')
            if not (folder / "grid.cuh").exists():
                (folder / "grid.cuh").write_text('#include <glass/ops.h>\n#include <vector>\n// header v1\n')
            return folder

        def command(self, folder, flags=("-O3", "-gencode", "arch=compute_120,code=sm_120"), compiler="nvcc"):
            return [compiler, *flags, f"-I{shared}", "-I", str(folder),
                    "-o", str(folder / "runner.exe"), str(folder / "runner.cu")]

        def run(self, name="a", **kw):
            folder = self.tree(name)
            return folder, cached_nvcc_run(self.command(folder, **kw), cwd=folder, what="fake")

        def builds(self):
            return len([line for line in log.read_text().splitlines() if line])

        def entries(self):
            root = tmp_path / "cache" / "commands"
            return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []

    return Kit()


def test_the_same_inputs_in_another_directory_are_a_hit(kit):
    first_dir, first = kit.run("a")
    second_dir, second = kit.run("b")
    assert first.returncode == 0 and second.returncode == 0
    assert kit.builds() == 1 and len(kit.entries()) == 1
    assert (second_dir / "runner.exe").read_bytes() == (first_dir / "runner.exe").read_bytes()
    assert second.stdout == first.stdout and second.stderr == first.stderr
    assert "warning: unused variable" in second.stderr
    assert os.access(second_dir / "runner.exe", os.X_OK)


@pytest.mark.parametrize("change", ["source", "local_header", "angle_header", "flag", "toolkit"])
def test_any_input_change_is_a_miss(kit, monkeypatch, change):
    kit.run("a")
    folder = kit.tree("b")
    flags = ("-O3", "-gencode", "arch=compute_120,code=sm_120")
    if change == "source":
        (folder / "runner.cu").write_text('#include "grid.cuh"\nint main() { return 1; }\n')
    elif change == "local_header":
        (folder / "grid.cuh").write_text('#include <glass/ops.h>\n// header v2\n')
    elif change == "angle_header":
        kit.library.write_text("// ops v2\n")
    elif change == "flag":
        flags = ("-O3", "-DEXTRA=1", "-gencode", "arch=compute_120,code=sm_120")
    else:
        monkeypatch.setenv("FAKE_NVCC_VERSION", "fake nvcc 2.0")
    result = cached_nvcc_run(kit.command(folder, flags=flags), cwd=folder)
    assert result.returncode == 0
    assert kit.builds() == 2 and len(kit.entries()) == 2


def test_the_output_path_and_include_paths_are_not_in_the_key(kit):
    a, b = kit.tree("a"), kit.tree("b")
    key_a = command_cache_key(kit.command(a), a)[0]
    key_b = command_cache_key(kit.command(b), b)[0]
    assert key_a == key_b
    relative = ["nvcc", "-O3", "-gencode", "arch=compute_120,code=sm_120", f"-I{kit.library.parent.parent}",
                "-I", ".", "-o", "runner.exe", "runner.cu"]
    assert command_cache_key(relative, a)[0] == key_a


def test_a_failed_command_is_returned_and_never_stored(kit, monkeypatch):
    monkeypatch.setenv("FAKE_NVCC_FAIL", "1")
    _, failed = kit.run("a")
    assert failed.returncode == 1 and "boom" in failed.stderr
    assert kit.entries() == []
    _, again = kit.run("a")
    assert again.returncode == 1 and kit.builds() == 2
    monkeypatch.delenv("FAKE_NVCC_FAIL")
    _, fixed = kit.run("a")
    assert fixed.returncode == 0 and kit.builds() == 3 and len(kit.entries()) == 1


def test_a_corrupt_entry_is_rebuilt(kit, monkeypatch):
    kit.run("a")
    entry = kit.root / "cache" / "commands" / kit.entries()[0]
    (entry / "output").write_text("corrupt")
    monkeypatch.setenv("FAKE_NVCC_STAMP", "rebuilt")
    folder, result = kit.run("b")
    assert result.returncode == 0 and kit.builds() == 2
    assert "rebuilt" in (folder / "runner.exe").read_text()
    assert "rebuilt" in (entry / "output").read_text()


def test_the_switch_and_an_unkeyable_command_run_unchanged(kit, monkeypatch):
    folder = kit.tree("a")
    no_output = ["nvcc", "-O3", str(folder / "runner.cu")]
    assert command_cache_key(no_output, folder) is None
    monkeypatch.setenv("GRID_CUDA_DISABLE_CACHE", "1")
    kit.run("a")
    kit.run("a")
    assert kit.builds() == 2 and kit.entries() == []


def test_two_builders_of_one_key_compile_once(kit, monkeypatch):
    monkeypatch.setenv("FAKE_NVCC_SLEEP", "0.4")
    folders = [kit.tree(name) for name in ("a", "b", "c")]
    results = []
    threads = [threading.Thread(target=lambda f=f: results.append(
        cached_nvcc_run(kit.command(f), cwd=f))) for f in folders]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.returncode for r in results] == [0, 0, 0]
    assert kit.builds() == 1
    assert len({(f / "runner.exe").read_bytes() for f in folders}) == 1


def _starts_with_nvcc(node) -> bool:
    import ast
    if isinstance(node, ast.BinOp):  # [nvcc, ...] + [...]
        return _starts_with_nvcc(node.left)
    if not isinstance(node, ast.List) or not node.elts:
        return False
    first = node.elts[0]
    return (isinstance(first, ast.Name) and first.id in ("nvcc", "compiler")) or (
        isinstance(first, ast.Constant) and first.value == "nvcc")


def _direct_nvcc_runs(path):
    """Line numbers of subprocess.run calls on an nvcc command in one test module."""
    import ast
    tree = ast.parse(path.read_text())
    found = []
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.Module)):
            continue
        commands = {node.targets[0].id for node in ast.walk(scope)
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and _starts_with_nvcc(node.value)}
        for node in ast.walk(scope):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("run", "check_call", "check_output", "Popen")
                    and isinstance(node.func.value, ast.Name) and node.args):
                first = node.args[0]
                if _starts_with_nvcc(first) or (isinstance(first, ast.Name) and first.id in commands):
                    found.append(node.lineno)
    return sorted(set(found))


def test_no_test_module_runs_nvcc_outside_the_cache():
    """A bare subprocess.run([nvcc, ...]) compiles on every run, inside the one
    GPU shard chain. Every test build goes through a cache helper."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    offenders = {}
    for folder in ("cuda_equivalents", "python_wrappers"):
        for module in sorted((root / folder).glob("test_*.py")):
            lines = _direct_nvcc_runs(module)
            if lines:
                offenders[f"test/{folder}/{module.name}"] = lines
    assert offenders == {}, f"use cached_nvcc_run for these compiles: {offenders}"


def test_no_test_module_compiles_a_varying_thread_count_into_its_program():
    """A thread count drawn per run and passed as -D gives the build a new key
    on every run, so the cache never hits. It goes to the runner at run time."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent
    varying = re.compile(r"-D\w*THREADS=\{")
    offenders = [f"test/{folder}/{module.name}"
                 for folder in ("cuda_equivalents", "python_wrappers")
                 for module in sorted((root / folder).glob("*.py"))
                 if varying.search(module.read_text())]
    assert offenders == [], f"pass the thread count to the runner as an argument: {offenders}"

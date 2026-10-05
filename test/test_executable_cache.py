"""CPU-only tests for the content-keyed CUDA test-executable cache.

A fake nvcc on PATH stands in for the toolkit: it writes an "executable" derived
from its source bytes and logs every invocation, so hits, rebuilds, failures and
concurrent builders are observable without a GPU or a real compile.
"""
from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from test.cuda_equivalents import cuda_harness
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

_FAKE_NVCC = """#!/usr/bin/env bash
if [ "$1" = "--version" ]; then echo "${FAKE_NVCC_VERSION:-fake nvcc 1.0}"; exit 0; fi
echo "$*" >> "$FAKE_NVCC_LOG"
[ -n "$FAKE_NVCC_SLEEP" ] && sleep "$FAKE_NVCC_SLEEP"
[ -n "$FAKE_NVCC_FAIL" ] && { echo "boom" >&2; exit 1; }
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
src="${@: -1}"
sha256sum "$src" grid.cuh > "$out"
"""


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
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
    for knob in ("FAKE_NVCC_FAIL", "FAKE_NVCC_SLEEP", "FAKE_NVCC_VERSION"):
        monkeypatch.delenv(knob, raising=False)

    src_dir = tmp_path / "src"
    inc_dir = tmp_path / "inc"
    src_dir.mkdir()
    inc_dir.mkdir()
    (src_dir / "runner.cu").write_text('#include "grid.cuh"\nint main() { return 0; }\n')
    (src_dir / "grid.cuh").write_text('#include "extra.cuh"\n// header v1\n')
    (inc_dir / "extra.cuh").write_text("// extra v1\n")

    class Toolkit:
        root = tmp_path
        runner = src_dir / "runner.cu"
        header = src_dir / "grid.cuh"
        extra = inc_dir / "extra.cuh"

        def builds(self):
            return len([line for line in log.read_text().splitlines() if line])

        def build(self, flags=("-O0",), variant=None, fallback="fallback"):
            return cached_nvcc_executable(
                [self.runner, self.header], list(flags), exe_name="runner.exe",
                fallback_dir=tmp_path / fallback, include_dirs=[inc_dir],
                variant=variant, what="fake runner")

        def entries(self):
            root = tmp_path / "cache" / "executables"
            return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []

    return Toolkit()


def test_second_build_is_a_hit(toolkit):
    exe1, cmd1 = toolkit.build()
    exe2, cmd2 = toolkit.build()
    assert toolkit.builds() == 1
    assert exe1 == exe2 and exe1.is_file()
    assert cmd1 == cmd2 and str(exe1) in cmd1
    assert len(toolkit.entries()) == 1


def test_inventory_uses_build_key_without_writes_or_compiles(toolkit):
    def inventory():
        return cached_nvcc_executable([toolkit.runner, toolkit.header], ['-O0'],
            exe_name='runner.exe', fallback_dir=toolkit.root / 'fallback',
            include_dirs=[toolkit.extra.parent], inspect_only=True)
    cold = inventory()
    assert cold['status'] == 'miss'
    assert toolkit.builds() == 0 and not (toolkit.root / 'cache').exists()
    exe, _ = toolkit.build()
    warm = inventory()
    assert warm['status'] == 'hit' and warm['key'] == cold['key'] == exe.parent.name
    exe.write_text('corrupt')
    assert inventory()['status'] == 'miss'
    assert toolkit.builds() == 1 and exe.read_text() == 'corrupt'


@pytest.mark.parametrize("change", ["header", "transitive_include", "runner"])
def test_any_input_byte_invalidates(toolkit, change):
    first, _ = toolkit.build()
    target = {"header": toolkit.header, "transitive_include": toolkit.extra,
              "runner": toolkit.runner}[change]
    target.write_text(target.read_text() + "// edited\n")
    second, _ = toolkit.build()
    assert toolkit.builds() == 2
    assert first != second and first.is_file()


def test_flags_variant_and_toolkit_are_in_the_key(toolkit, monkeypatch):
    toolkit.build(flags=["-O0"])
    toolkit.build(flags=["-O0", "-DGRID_DEFAULT_RESOURCE_TIER=TIER_LITE"])
    toolkit.build(flags=["-O0"], variant={"scalar": "double"})
    monkeypatch.setenv("FAKE_NVCC_VERSION", "fake nvcc 2.0")
    toolkit.build(flags=["-O0"])
    assert toolkit.builds() == 4
    assert len(toolkit.entries()) == 4


def test_key_ignores_the_directory_the_sources_came_from(toolkit, tmp_path):
    toolkit.build()
    other = tmp_path / "elsewhere"
    other.mkdir()
    for path in (toolkit.runner, toolkit.header):
        (other / path.name).write_bytes(path.read_bytes())
    toolkit.runner, toolkit.header = other / "runner.cu", other / "grid.cuh"
    # extra.cuh is still reached through the include dir, with the same bytes.
    toolkit.build()
    assert toolkit.builds() == 1


def test_failed_build_is_never_published(toolkit, monkeypatch):
    monkeypatch.setenv("FAKE_NVCC_FAIL", "1")
    with pytest.raises(pytest.fail.Exception, match="boom"):
        toolkit.build()
    assert toolkit.entries() == []
    monkeypatch.delenv("FAKE_NVCC_FAIL")
    exe, _ = toolkit.build()
    assert exe.is_file() and toolkit.builds() == 2


def test_corrupted_entry_is_rebuilt(toolkit):
    exe, _ = toolkit.build()
    exe.write_text("truncated")
    rebuilt, _ = toolkit.build()
    assert toolkit.builds() == 2
    assert rebuilt == exe and rebuilt.read_text() != "truncated"


def test_entry_without_manifest_is_rebuilt(toolkit):
    exe, _ = toolkit.build()
    (exe.parent / "manifest.json").unlink()
    toolkit.build()
    assert toolkit.builds() == 2


def test_concurrent_builders_compile_once(toolkit, monkeypatch):
    monkeypatch.setenv("FAKE_NVCC_SLEEP", "0.5")
    results, errors = [], []

    def worker():
        try:
            results.append(toolkit.build()[0])
        except BaseException as exc:  # surface a failure from the thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert toolkit.builds() == 1
    assert len(set(results)) == 1


def test_disabled_cache_builds_in_the_fallback_dir(toolkit, monkeypatch):
    monkeypatch.setenv("GRID_CUDA_DISABLE_CACHE", "1")
    exe, _ = toolkit.build()
    assert exe == toolkit.root / "fallback" / "runner.exe"
    assert toolkit.entries() == []


def test_manifest_records_provenance(toolkit):
    import json
    exe, _ = toolkit.build()
    manifest = json.loads((exe.parent / "manifest.json").read_text())
    assert manifest["key"] == exe.parent.name
    assert manifest["executable_sha256"] == cuda_harness._hash_file(exe)
    names = [name for name, _ in manifest["key_payload"]["sources"]]
    assert names == sorted(["runner.cu", "grid.cuh", "extra.cuh"])
    assert manifest["key_payload"]["toolkit"]["version"] == "fake nvcc 1.0"

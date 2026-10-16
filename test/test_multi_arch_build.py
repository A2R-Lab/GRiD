"""Opt-in multi-architecture build (GRID_RBD_CUDA_ARCHS, 2026-10-15), CPU-only.

The env var names a comma list of compute capabilities; compile_so then emits
one -gencode per arch (plus the build's own cuda_arch), the cache keys fold the
sorted list so a fat build never collides with a single-arch one, the build
identity records it as ``cuda_archs``, and a load accepts a stored entry whose
``cuda_archs`` contains the detected arch. Unset keeps the single-arch build
byte for byte. nvcc is never run here: the nvcc command is captured at the
``_build_units`` boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grid_rbd import _cache, _compile

ENV = _cache.CUDA_ARCHS_ENV
URDF = b"<robot name='fat_probe'><link name='base'/></robot>"


@pytest.fixture(autouse=True)
def _single_arch_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


# ─── (a) the nvcc command ───────────────────────────────────────────────────

def _assemble(monkeypatch, tmp_path, cuda_arch=120):
    """Run compile_so up to the point nvcc would be invoked; return what it
    assembled (the full flag list and the arch label)."""
    captured = {}

    def fake_build_units(cmd, source_dir, out_so, arch, core_flags=None,
                         object_cache_dir=None, reuse_objects=True):
        captured["cmd"] = list(cmd)
        captured["arch"] = arch

    monkeypatch.setattr(_compile, "_build_units", fake_build_units)
    monkeypatch.setattr(_compile, "find_nvcc", lambda: "nvcc")
    monkeypatch.setattr(_compile, "_jax_ffi_include_dir", lambda: None)
    monkeypatch.setattr(_compile, "_torch_build_flags", lambda: None)
    wrapper = tmp_path / "wrapper.cu"
    wrapper.write_text("// wrapper\n")
    _compile.compile_so(wrapper, tmp_path / "robot.so", cuda_arch=cuda_arch)
    assert "cmd" in captured, "compile_so never reached the nvcc stage"
    return captured


def _gencode(cmd):
    return [a for a in cmd if a.startswith("-gencode=")]


def test_unset_keeps_the_single_arch_command(monkeypatch, tmp_path):
    got = _assemble(monkeypatch, tmp_path, cuda_arch=120)
    assert _gencode(got["cmd"]) == ["-gencode=arch=compute_120,code=sm_120"]
    assert "-DGRID_RBD_ARCH=120" in got["cmd"]
    assert not [a for a in got["cmd"] if a.startswith("-DGRID_RBD_ARCHS=")]
    assert got["arch"] == "sm_120"


def test_fat_command_has_one_gencode_per_arch_sorted_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "120,86,86")
    got = _assemble(monkeypatch, tmp_path, cuda_arch=120)
    assert _gencode(got["cmd"]) == [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    # the tuning arch is unchanged; the embedded list is named for the wrapper
    assert "-DGRID_RBD_ARCH=120" in got["cmd"]
    assert "-DGRID_RBD_ARCHS=86:120" in got["cmd"]
    assert got["arch"] == "sm_86+sm_120"


def test_fat_command_always_embeds_the_tuning_arch(monkeypatch, tmp_path):
    """The arch the launch configs were selected for is always in the binary,
    even when the env list omits it (the local GPU must run what it tuned)."""
    monkeypatch.setenv(ENV, "86,89")
    got = _assemble(monkeypatch, tmp_path, cuda_arch=120)
    assert _gencode(got["cmd"]) == [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    assert "-DGRID_RBD_ARCH=120" in got["cmd"]
    assert "-DGRID_RBD_ARCHS=86:89:120" in got["cmd"]


# ─── (b) cache keys ──────────────────────────────────────────────────────────

def _source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir(parents=True)
    (d / "grid.cuh").write_text("// grid.cuh\n")
    (d / "wrapper.cu").write_text("// wrapper\n")
    (d / "wrapper_common.cuh").write_text("// common\n")
    return d


def test_keys_differ_between_single_and_fat_and_are_order_stable(monkeypatch, tmp_path):
    src = _source_dir(tmp_path)

    def keys():
        return (_cache.compute_cache_key(URDF, {}, 120),
                _cache.compute_content_key(src, {}, 120, 256))

    single = keys()
    monkeypatch.setenv(ENV, "86,120")
    fat = keys()
    monkeypatch.setenv(ENV, "120,86,86")
    fat_reordered = keys()
    monkeypatch.setenv(ENV, "86")   # the tuning arch is folded in: same embedded set
    fat_implicit = keys()
    monkeypatch.setenv(ENV, "86,89,120")
    fat_more = keys()
    assert single[0] != fat[0] and single[1] != fat[1]
    assert fat == fat_reordered == fat_implicit
    assert fat_more != fat


def test_identity_records_cuda_archs_only_when_set(monkeypatch):
    assert "cuda_archs" not in _cache.build_identity(120)
    monkeypatch.setenv(ENV, "120,86")
    ident = _cache.build_identity(120)
    assert ident["cuda_arch"] == 120 and ident["cuda_archs"] == [86, 120]


# ─── (c) identity acceptance at load ────────────────────────────────────────

def _entry(tmp_path: Path, record: dict) -> Path:
    d = tmp_path / "entry"
    d.mkdir()
    _cache.write_build_inputs(d, record)
    return d


def test_fat_entry_loads_on_an_embedded_arch_and_refuses_another(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "86,120")
    recorded = _cache.build_identity(120)            # compiled with sm_120 tuning
    monkeypatch.delenv(ENV)                          # the loading box sets nothing
    entry = _entry(tmp_path, recorded)
    assert _cache.load_incompat_reasons(entry, _cache.build_identity(86)) == []
    assert _cache.load_incompat_reasons(entry, _cache.build_identity(120)) == []
    assert _cache.load_incompat_reasons(entry, _cache.build_identity(75)) == [
        "cuda_arch: recorded 120 != current 75"]


def test_single_arch_entry_still_refuses_another_arch(tmp_path):
    entry = _entry(tmp_path, _cache.build_identity(120))
    assert _cache.load_incompat_reasons(entry, _cache.build_identity(86)) == [
        "cuda_arch: recorded 120 != current 86"]


# ─── (d) the env var is validated, never silently ignored ───────────────────

@pytest.mark.parametrize("value", ["sm_120", "", "8.6", "86,,"])
def test_malformed_env_is_rejected_with_the_variable_named(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        _cache.cuda_archs_from_env()
    with pytest.raises(ValueError, match=ENV):
        _cache.compute_cache_key(URDF, {}, 120)


def test_env_parses_to_a_sorted_deduplicated_list(monkeypatch):
    assert _cache.cuda_archs_from_env() is None
    monkeypatch.setenv(ENV, " 120, 86 ,86,90")
    assert _cache.cuda_archs_from_env() == [86, 90, 120]
    assert _cache.effective_cuda_archs(89) == [86, 89, 90, 120]
    assert _cache.effective_cuda_archs(89, [120]) == [89, 120]


# ─── the meta record ────────────────────────────────────────────────────────

def _compile_sources(monkeypatch, tmp_path, cuda_arch=120):
    d = _source_dir(tmp_path)
    monkeypatch.setattr(_compile, "compile_so", lambda *a, **k: None)
    monkeypatch.setattr(_compile, "_mjx_signature_flags", lambda cuh: [])
    meta = _compile.compile_sources(d, {"NUM_JOINTS": 1}, {}, cuda_arch=cuda_arch, max_batch=8)
    assert json.loads((d / "meta.json").read_text()) == meta
    return meta


def test_meta_records_cuda_archs_and_the_tuning_arch_only_when_set(monkeypatch, tmp_path):
    meta = _compile_sources(monkeypatch, tmp_path / "single")
    assert meta["cuda_arch"] == 120 and "cuda_archs" not in meta and "cuda_archs_tuning" not in meta
    monkeypatch.setenv(ENV, "86,89,90,120")
    meta = _compile_sources(monkeypatch, tmp_path / "fat")
    assert meta["cuda_arch"] == 120 and meta["cuda_archs"] == [86, 89, 90, 120]
    assert "sm_120" in meta["cuda_archs_tuning"]

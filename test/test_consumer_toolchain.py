"""Consumer-only compiler selection and visibility policy, without compilation."""
from types import SimpleNamespace

import pytest

from grid_rbd import _consumer_toolchain as tc


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    host = tmp_path / "g++"
    host.write_bytes(b"fake host")
    for key in tc._ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tc, "_resolve_host", lambda selected: str(host) if not selected else selected)
    monkeypatch.setattr(tc.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        stdout="#define __GNUC__ 13\n" if "-dM" in cmd else "test compiler"))
    return host


def test_default_gcc_isolation_and_pinned_link(compiler):
    result = tc.resolve_consumer_toolchain([], [])
    assert "-fno-gnu-unique" in result.isolation[0]
    assert "-fvisibility=hidden" in result.isolation[0]
    assert f"-ccbin={compiler}" in result.compile_flags()
    assert f"-ccbin={compiler}" in result.link_command("nvcc", "out.so", ["unit.o"])


def test_clang_does_not_receive_gcc_only_flag(compiler, monkeypatch):
    monkeypatch.setattr(tc.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="#define __clang__ 1\n#define __GNUC__ 4\n"))
    assert "-fno-gnu-unique" not in tc.resolve_consumer_toolchain([], []).isolation[0]


def test_flags_and_environment_follow_one_host(compiler, monkeypatch):
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DENV_PRE=1 -ccbin=/ignored")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", f"--compiler-bindir={compiler} -DENV_POST=2")
    result = tc.resolve_consumer_toolchain(["-ccbin", "/also-ignored", "-DUSER=3"], ["-lm"])
    assert result.host == str(compiler)
    assert result.prepend == ["-DENV_PRE=1"]
    assert result.append == ["-DENV_POST=2"]
    assert "-lm" in result.link_command("nvcc", "out", [])
    assert not any(key in result.environment() for key in tc._ENV_KEYS)
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DCHANGED=1")
    with pytest.raises(RuntimeError, match="environment changed"):
        result.environment()


def test_mismatched_link_host_is_rejected(compiler):
    with pytest.raises(ValueError, match="same host"):
        tc.resolve_consumer_toolchain([f"-ccbin={compiler}"], ["-ccbin=/different"])


@pytest.mark.parametrize("flags", [["-optf", "file"], ["--options-file=file"], ["@file"], ["-ccbin"]])
def test_untracked_or_missing_flag_inputs_fail(compiler, flags):
    with pytest.raises(ValueError):
        tc.resolve_consumer_toolchain(flags, [])


def test_host_bytes_participate_in_identity(compiler):
    before = tc.resolve_consumer_toolchain([], [])
    compiler.write_bytes(b"replacement host")
    after = tc.resolve_consumer_toolchain([], [])
    assert before.identity["host_sha256"] != after.identity["host_sha256"]

"""Resolve one explicit host compiler and consumer-only symbol policy.

nvcc environment flags are flattened once, then removed from child environments
so compile and link use the same recorded selection. Wrapper builds do not use
this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess


_ENV_KEYS = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN")


def _host_selection(flags):
    """Remove nvcc host selection arguments, retaining their last explicit value."""
    remaining, selected = [], None
    iterator = iter(flags)
    for flag in iterator:
        if flag in ("--options-file", "-optf") or flag.startswith(("--options-file=", "-optf=", "@")):
            raise ValueError("consumer builds require explicit flags, not nvcc options files; "
                             "expand the file into extra_flags/link_flags for recorded identity")
        if flag in ("-ccbin", "--compiler-bindir"):
            selected = next(iterator, None)
            if not selected or selected.startswith("-"):
                raise ValueError(f"{flag} requires a host compiler path")
        elif flag.startswith(("-ccbin=", "--compiler-bindir=")):
            selected = flag.split("=", 1)[1]
            if not selected:
                raise ValueError("empty host compiler selection")
        else:
            remaining.append(flag)
    return remaining, selected


def _resolve_host(selected):
    selected = selected or "g++"
    if Path(selected).is_dir():
        selected = str(Path(selected) / "g++")
    path = shutil.which(selected)
    if path is None:
        raise RuntimeError(f"consumer host C++ compiler not found: {selected}")
    # Preserve the driver spelling: clang++ and clang may be symlinks to the
    # same binary but select different link behavior from argv[0].
    return str(Path(path).absolute())


@dataclass(frozen=True)
class ConsumerToolchain:
    host: str
    prepend: list[str]
    append: list[str]
    extra: list[str]
    link: list[str]
    isolation: list[str]
    identity: dict
    env_flags: dict

    def environment(self):
        env = dict(os.environ)
        if {key: env.get(key, "") for key in _ENV_KEYS} != self.env_flags:
            raise RuntimeError("nvcc environment changed during consumer build")
        for key in _ENV_KEYS:
            env.pop(key, None)
        return env

    def compile_flags(self):
        return self.extra + self.append + [f"-ccbin={self.host}"] + self.isolation

    def link_command(self, nvcc, out, objects):
        return ([nvcc] + self.prepend + ["--shared", "-o", str(out)] + objects +
                self.link + self.append + [f"-ccbin={self.host}"])


def resolve_consumer_toolchain(extra_flags, link_flags):
    env = {key: os.environ.get(key, "") for key in _ENV_KEYS}
    pre, pre_host = _host_selection(shlex.split(env["NVCC_PREPEND_FLAGS"]))
    post, post_host = _host_selection(shlex.split(env["NVCC_APPEND_FLAGS"]))
    extra, extra_host = _host_selection(extra_flags)
    link, link_host = _host_selection(link_flags)
    host = _resolve_host(post_host or extra_host or pre_host or env["NVCC_CCBIN"])
    explicit_link_host = post_host or link_host
    if explicit_link_host and _resolve_host(explicit_link_host) != host:
        raise ValueError("consumer compile and link must select the same host compiler")
    macros = subprocess.run([host, "-dM", "-E", "-x", "c++", "-"], input="",
                            capture_output=True, text=True, check=True, timeout=30).stdout
    if "#define __clang__ " in macros:
        family = "clang"
    elif "#define __GNUC__ " in macros:
        family = "gcc"
    else:
        raise RuntimeError("consumer symbol isolation supports GCC and Clang host compilers only")
    flags = "-fvisibility=hidden,-fvisibility-inlines-hidden"
    if family == "gcc":
        flags += ",-fno-gnu-unique"
    version = subprocess.run([host, "--version"], capture_output=True, text=True,
                             check=True, timeout=30).stdout
    identity = {"host": host, "host_sha256": hashlib.sha256(Path(host).read_bytes()).hexdigest(),
                "host_version": version, "family": family, "nvcc_environment": env,
                "isolation": flags}
    return ConsumerToolchain(host, pre, post, extra, link,
                             ["--compiler-options=" + flags], identity, env)

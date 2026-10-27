"""Non-system CUDA dependencies for consumer builds, never prepared loads.

Every lookup rediscovers selection, including new earlier include-path matches.
System development headers excluded by nvcc -MM remain an environment assumption.
Inputs must stay immutable during discovery/build; this is not a filesystem snapshot.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

DEPENDENCY_SCHEMA = 1
_TARGET = "grid_dependency"


def parse_dependencies(output: str) -> list[Path]:
    """Parse nvcc's Make-like rule; it escapes spaces but leaves # and $ literal.

    This is compiler output, not a Make program. Never expand shell/Make syntax.
    """
    text = re.sub(r"\\\r?\n", " ", output).strip()
    target = re.match(re.escape(_TARGET) + r"[ \t]*:", text)
    if target is None:
        raise RuntimeError("consumer dependency output lacks the requested Make target")
    text = text[target.end():]
    tokens, word = [], []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 1
            if i == len(text):
                raise RuntimeError("consumer dependency output has a dangling escape")
            if text[i].isspace():
                word.append(text[i])
            else:
                word.extend(("\\", text[i]))
        elif ch in "\r\n":
            raise RuntimeError("consumer dependency output has multiple rules")
        elif ch.isspace():
            if word:
                tokens.append("".join(word)); word = []
        else:
            word.append(ch)
        i += 1
    if word:
        tokens.append("".join(word))
    if not tokens:
        raise RuntimeError("consumer dependency output has no files")
    # Preserve compiler spelling: two symlink aliases to identical bytes can
    # expand __FILE__ differently. Record the resolved target separately below.
    return sorted({Path(token).absolute() for token in tokens})


def discover_dependencies(gen_dir, sources, compile_cmd, toolchain=None):
    """Return per-unit path/digest records with generated paths relocation-neutral."""
    # The common compile command contains semantic flags, not phase/output options.
    # Refuse overrides rather than silently preprocess a different compilation.
    forbidden = {"-M", "-MM", "-MD", "-MMD", "-MP", "-MF", "-MT", "-MQ",
                 "-o", "-c", "-E", "--compile", "--preprocess", "--shared", "--lib",
                 "--ptx", "--cubin", "--fatbin", "--run", "--output-file",
                 "--dependency-output", "--dependency-target-name",
                 "--generate-dependencies", "--generate-nonsystem-dependencies",
                 "--generate-dependencies-with-compile",
                 "--generate-nonsystem-dependencies-with-compile"}
    for flag in compile_cmd[1:]:
        if flag.split("=", 1)[0] in forbidden or any(
                flag.startswith(prefix) and flag != prefix for prefix in ("-MF", "-MT", "-MQ")):
            raise ValueError(f"consumer dependency discovery does not accept phase/output option {flag!r}")
    generated = Path(gen_dir).resolve()
    units = {}
    for source in sources:
        source = Path(source).resolve()
        command = [*compile_cmd, "-MM", "-MT", _TARGET, str(source)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                    env=toolchain.environment() if toolchain else None)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"consumer dependency discovery failed for {source}: {exc}") from exc
        if result.returncode:
            raise RuntimeError(f"consumer dependency discovery failed for {source} (nvcc exit "
                               f"{result.returncode}):\n{result.stdout}\n{result.stderr}")
        paths = parse_dependencies(result.stdout)
        if source not in paths:
            raise RuntimeError(f"consumer dependency output omits source {source}")
        def path_name(path):
            try:
                return "@generated/" + path.relative_to(generated).as_posix()
            except ValueError:
                return str(path)

        rows = []
        for path in paths:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(f"consumer dependency cannot be read: {path}; "
                                   "nvcc dependency output must name readable files "
                                   "(literal backslashes in POSIX filenames are not preserved by nvcc)") from exc
            rows.append({"path": path_name(path), "resolved_path": path_name(path.resolve()),
                         "sha256": digest})
        units[str(source)] = sorted(rows, key=lambda row: row["path"])
    return {"schema": DEPENDENCY_SCHEMA, "mode": "nvcc-MM", "units": units}

"""Is the installed grid_rbd the one this checkout's sources describe?

`grid_rbd._core` is a compiled extension and `pip install -e .` builds it once. Pulling new
sources does not rebuild it, so a checkout can carry an extension that predates methods the
tests call. A receipt refresh on such a checkout ran for seven hours and then failed 33
tests with AttributeError (2026-11-21). The launcher runs this check before it starts.

Run as a script it exits nonzero and names what is stale.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BOUND = re.compile(r'\.def(?:_static|_property_readonly|_property|_readwrite|_readonly)?\(\s*"([A-Za-z_][A-Za-z_0-9]*)"')
_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def bound_names(core_source):
    """Every name the pybind source binds, on the module or on one of its classes."""
    return set(_BOUND.findall(core_source))


def exposed_names(module):
    names = set(dir(module))
    for value in vars(module).values():
        if isinstance(value, type):
            names |= set(dir(value))
    return names


def problems(core_source, pyproject_text, module, installed_version):
    found = []
    missing = sorted(bound_names(core_source) - exposed_names(module))
    if missing:
        found.append(f"grid_rbd._core ({getattr(module, '__file__', '?')}) lacks {len(missing)} "
                     f"binding(s) that bindings/src/_core.cpp declares: {', '.join(missing[:8])}"
                     + (" ..." if len(missing) > 8 else ""))
    declared = _VERSION.search(pyproject_text)
    if declared and declared.group(1) != installed_version:
        found.append(f"installed grid-rbd is {installed_version}, pyproject.toml says {declared.group(1)}")
    return found


def main():
    import grid_rbd
    import grid_rbd._core as core
    found = problems((ROOT / "bindings/src/_core.cpp").read_text(), (ROOT / "pyproject.toml").read_text(),
                     core, grid_rbd.__version__)
    for line in found:
        print(line)
    if found:
        print("Rebuild the editable install: .venv/bin/python -m pip install -e . --no-deps (make build).")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

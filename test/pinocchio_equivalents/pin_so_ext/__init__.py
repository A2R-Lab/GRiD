"""Loader for the Pinocchio second-order RNEA Python extension.

The C++ source lives next to this file (`pin_so_ext.cpp`). The compiled extension
is built into this same directory via `python setup.py build_ext --inplace`. The
`load()` helper compiles on demand the first time it's called, so a fresh checkout
can run the tests without manual setup as long as pybind11, g++, and the Pinocchio
development headers are available.
"""
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _find_compiled():
    for entry in _DIR.iterdir():
        if entry.name.startswith("pin_so_ext.") and entry.name.endswith(".so"):
            return entry
    return None


def _build():
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    subprocess.check_call(cmd, cwd=_DIR)


def load():
    """Return the compiled `pin_so_ext` module, building it on first use."""
    so = _find_compiled()
    if so is None:
        _build()
        so = _find_compiled()
        if so is None:
            raise RuntimeError(
                "Failed to build pin_so_ext. Ensure pybind11 is installed and "
                "Pinocchio development headers are available (pkg-config pinocchio)."
            )
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))
    return importlib.import_module("pin_so_ext")

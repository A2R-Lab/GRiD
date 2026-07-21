"""Root build shim for grid-rbd's pybind11 Runner extension.

All project metadata lives in pyproject.toml; this file exists only to declare
the C++ extension (setuptools has no pyproject-native way to declare a pybind11
Extension). Most of the package is pure Python — the codegen toolkit plus the
runtime wrapper's cache management + nvcc invocation at register_robot time.
The _core extension is a small shim that dlopens the per-robot .so (built at
register_robot time) and dispatches numpy<->ctypes-friendly calls to it.

Building this extension at pip-install time requires only a C++17 compiler.
nvcc is NOT needed for `pip install -e .` — it comes into play only later, when
the user calls register_robot().
"""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "grid_rbd._core",
        sources=["bindings/src/_core.cpp"],
        cxx_std=17,
        # The Runner dlopens the per-robot .so; needs to link libdl on Linux.
        # Windows / macOS use different mechanisms but we're Linux-only in v1
        # (CUDA's primary platform).
        libraries=["dl"],
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

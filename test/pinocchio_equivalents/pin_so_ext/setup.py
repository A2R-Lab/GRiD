"""Build script for the Pinocchio second-order RNEA binding.

Note: pinocchio.pc lives inside the venv's cmeel.prefix
(``<venv>/lib/python*/site-packages/cmeel.prefix/lib/pkgconfig``), not on
the system pkg-config path. The repo's ``developer_install.sh`` exports
``PKG_CONFIG_PATH`` to that location before invoking this script; if you
run setup.py standalone you must export it yourself or pkg-config will
fail to find ``pinocchio.pc``.
"""
import os
import subprocess
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _pin_pkg_config():
    raw = subprocess.check_output(["pkg-config", "--cflags", "--libs", "pinocchio"], text=True)
    include_dirs, library_dirs, libraries, defines, extra_link = [], [], [], [], []
    for token in raw.split():
        if token.startswith("-I"):
            include_dirs.append(token[2:])
        elif token.startswith("-L"):
            library_dirs.append(token[2:])
        elif token.startswith("-l"):
            libraries.append(token[2:])
        elif token.startswith("-D"):
            key_val = token[2:].split("=", 1)
            defines.append((key_val[0], key_val[1] if len(key_val) > 1 else None))
        elif token.startswith("-Wl,-rpath,"):
            extra_link.append(token)
    return include_dirs, library_dirs, libraries, defines, extra_link


include_dirs, library_dirs, libraries, defines, extra_link = _pin_pkg_config()

ext = Pybind11Extension(
    "pin_so_ext",
    sources=["pin_so_ext.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=defines,
    extra_link_args=extra_link,
    cxx_std=17,
)

setup(
    name="pin_so_ext",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

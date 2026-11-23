"""The one nvcc flag list for code that compiles a generated GRiD header.

The Python wrapper (bindings/grid_rbd/_compile.py), the native CUDA tests
(test/cuda_equivalents), the examples and the documented commands all build
their command lines from this module, so the kernels a test validates are
compiled the way the kernels an installed user runs are compiled.

Why the list matters: ``-ftz``, ``-prec-div``, ``-prec-sqrt`` and ``-fmad``
change the device code's floating point (flush-to-zero, reciprocal-based
division, approximate square root, fused multiply-add contraction). nvcc's
``-O`` level only reaches host code, but it is part of the list so every
build passes the same one. ``-fmad=true`` is nvcc's default and is stated
explicitly so the command line records it.

API:

``KERNEL_FLAGS``
    The default tuple, in command-line order.
``kernel_flags(environ=None)``
    The list to put on an nvcc command line (a fresh ``list[str]``). Honours
    the ``GRID_NVCC_FP_FLAGS`` override. Call it when the command is built,
    not at import, so the override is read at the time of the build.
``fp_flags(environ=None)``
    Only the four floating-point flags, after the override.
``CXX_STANDARD_FLAG``, ``OPTIMIZATION_FLAG``, ``DEFAULT_FP_FLAGS``, ``IEEE_FP_FLAGS``
    The parts of the list.

Command line: ``python -m grid_codegen.nvcc_flags`` (or ``grid-generate
--print-nvcc-flags``) prints the list on one line, for use in a shell:
``nvcc -arch=sm_90 $(python -m grid_codegen.nvcc_flags) my_kernel.cu``.

Override, for experiments only: ``GRID_NVCC_FP_FLAGS`` replaces the four
floating-point flags and nothing else.

* unset, empty or ``default``: ``DEFAULT_FP_FLAGS``.
* ``ieee``: ``-ftz=false -prec-div=true -prec-sqrt=true -fmad=true``
  (denormals kept, IEEE division and square root; contraction stays on).
* a space-separated list of ``-ftz=``, ``-prec-div=``, ``-prec-sqrt=`` and
  ``-fmad=`` settings, each ``true`` or ``false``. A flag the list leaves out
  keeps its default value, so the command always states all four. Any other
  token raises ``ValueError``.

Caches: every cache that stores a compiled binary keys on the resolved list,
so a changed override never reuses a binary built under another one. The
wrapper's per-unit object key hashes the compile command, its store keys fold
``kernel_flags()`` (bindings/grid_rbd/_cache.py), and the native test caches
hash the flags passed to them (test/cuda_equivalents/cuda_harness.py and
executable_cache.py). The name of the variable is ``env_knobs.COMPILE_ENV_KNOBS``.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping

from .env_knobs import COMPILE_ENV_KNOBS

FP_FLAGS_ENV = COMPILE_ENV_KNOBS[0]

CXX_STANDARD_FLAG = "-std=c++17"
OPTIMIZATION_FLAG = "-O3"
DEFAULT_FP_FLAGS = ("-ftz=true", "-prec-div=false", "-prec-sqrt=false", "-fmad=true")
IEEE_FP_FLAGS = ("-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=true")
KERNEL_FLAGS = (CXX_STANDARD_FLAG, OPTIMIZATION_FLAG, *DEFAULT_FP_FLAGS)

_FP_NAMES = tuple(flag.split("=", 1)[0] for flag in DEFAULT_FP_FLAGS)


def fp_flags(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The four floating-point flags, after the ``GRID_NVCC_FP_FLAGS`` override."""
    raw = (os.environ if environ is None else environ).get(FP_FLAGS_ENV, "")
    spec = raw.strip()
    if spec.lower() in ("", "default"):
        return DEFAULT_FP_FLAGS
    if spec.lower() == "ieee":
        return IEEE_FP_FLAGS
    chosen = dict(flag.split("=", 1) for flag in DEFAULT_FP_FLAGS)
    seen = set()
    for token in spec.split():
        name, sep, value = token.partition("=")
        if not sep or name not in _FP_NAMES or value not in ("true", "false") or name in seen:
            raise ValueError(
                f"{FP_FLAGS_ENV}={raw!r}: {token!r} is not accepted. Use 'default', 'ieee', or a "
                f"space-separated list of {', '.join(n + '=true|false' for n in _FP_NAMES)} "
                "with each flag at most once.")
        seen.add(name)
        chosen[name] = value
    return tuple(f"{name}={chosen[name]}" for name in _FP_NAMES)


def kernel_flags(environ: Mapping[str, str] | None = None) -> list[str]:
    """The nvcc flags every build of a generated header passes, in order."""
    return [CXX_STANDARD_FLAG, OPTIMIZATION_FLAG, *fp_flags(environ)]


def main(argv: list[str] | None = None) -> int:
    """Print the list on one line (``python -m grid_codegen.nvcc_flags``)."""
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("usage: python -m grid_codegen.nvcc_flags\n"
              f"Prints the nvcc flags GRiD compiles generated headers with. {FP_FLAGS_ENV} "
              "(default, ieee, or an explicit list) replaces the floating-point flags.",
              file=sys.stderr if args[0] not in ("-h", "--help") else sys.stdout)
        return 0 if args[0] in ("-h", "--help") else 2
    print(" ".join(kernel_flags()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

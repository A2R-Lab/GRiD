"""Runtime canary for `abi_specs.cabi_direct` rows (2026-10-02).

A direct row aims the generated host wrapper's device->host copy at the CALLER's buffer.
The CPU referee (test/test_cabi_direct_mirror_sizes.py) proves the copy size statically;
this is the same claim measured on a real artifact: call the C ABI with a buffer that is
followed by guard words and check that every output element was written and that no
guard word was. Two defects of this feature got past static reasoning: twins that wrote
nothing, and a named-target build whose download was 4x the public size. The first kind
fails here on any artifact with twins; the second only on a build configuration the
calling test registers, so the static rule (no wrapper-side macro in a direct row's size)
stays the guard for configurations no test builds.
"""
from __future__ import annotations

import ctypes
import re
from pathlib import Path

import numpy as np

from grid_codegen.abi_specs import ABI_SPECS

_GUARD = np.uint32(0xDEADBEEF)
_PAD = 4096                      # guard words after the output


def _header_constants(handle):
    src = (Path(handle._so_path).parent / "grid.cuh").read_text()
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"const int (\w+) = (\d+);", src)}


def direct_keys():
    return sorted(k for k, s in ABI_SPECS.items() if s.cabi_direct)


def guarded_call(handle, key, q, qd, u, *, mjx=False):
    """Call grid_rbd_<key>[_mujoco] straight through the C ABI. Returns
    (rc, unwritten_output_words, overwritten_guard_words, output)."""
    spec = ABI_SPECS[key]
    consts = _header_constants(handle)
    size = int(eval(spec.out_size_expr.replace("grid::", ""), {"__builtins__": {}}, consts))
    batch = q.shape[0]
    words = np.full(batch * size + _PAD, _GUARD, np.uint32)
    fp = ctypes.POINTER(ctypes.c_float)
    values, types = [ctypes.c_longlong(handle.ctx_id)], [ctypes.c_longlong]
    for name, ctype in spec.inputs:
        if name == "batch":
            values.append(ctypes.c_int(batch)); types.append(ctypes.c_int)
        elif name == "it":
            values.append(ctypes.c_int(0)); types.append(ctypes.c_int)          # Euler
        elif name in ("gravity", "dt"):
            values.append(ctypes.c_float(-9.81 if name == "gravity" else 0.01)); types.append(ctypes.c_float)
        elif name == "f_ext":
            values.append(None); types.append(fp)
        elif ctype == "T*":
            values.append(words.ctypes.data_as(fp)); types.append(fp)
        else:
            array = {"q": q, "qd": qd}.get(name, u)                              # u / qdd / qdd_opt
            values.append(np.ascontiguousarray(array, np.float32).ctypes.data_as(fp)); types.append(fp)
    fn = getattr(ctypes.CDLL(str(handle._so_path)), "grid_rbd_" + (spec.abi_stem or key) + ("_mujoco" if mjx else ""))
    fn.argtypes, fn.restype = types, ctypes.c_int
    rc = fn(*values)
    out, guard = words[:batch * size], words[batch * size:]
    return rc, int((out == _GUARD).sum()), int((guard != _GUARD).sum()), out.view(np.float32).reshape(batch, size)

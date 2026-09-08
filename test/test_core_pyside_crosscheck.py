"""C4 slice-1 referee: AbiSpec py_* fields vs the hand-written _core.cpp.

Until the C4 generator emits the pybind method bodies from ABI_SPECS, both the
table and the hand code exist side by side — this cross-check keeps the
transcription honest (same contract as test_abi_spec_crosscheck.py plays for
wrapper_template.cu). It parses bindings/src/_core.cpp and validates, for every
spec row with a pybind method:

- ``py_out_dims`` == the dims the method's ``py::array_t<CT> out({batch, ...})``
  allocates (verbatim C++ exprs);
- ``py_rc3_msg``  == the method's ``rc == 3`` message (adjacent C string
  literals joined);
- ``py_twin_guard`` == the ``*_mujoco`` twin's null-fn guard message.

When the C4 generator lands, this file is superseded by the generated-region
drift gate and should be retired with it.
"""
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_CORE = _REPO / "bindings" / "src" / "_core.cpp"

_METHOD = re.compile(
    r"\n    py::array_t<CT> (\w+)\((.*?)\)\n?\s*\{\n(.*?)\n    \}\n", re.DOTALL)
_OUT = re.compile(r"py::array_t<CT> out\(\{batch,?\s*([^}]*)\}\)")
_RC3 = re.compile(r"if \(rc == 3\) throw std::runtime_error\((.*?)\);\n", re.DOTALL)
_GUARD = re.compile(r"if \(!fn_\w+\)\s*\{?\s*\n?\s*throw std::runtime_error\((.*?)\);", re.DOTALL)


def _cjoin(frag: str) -> str:
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', frag)
    return "".join(p.replace('\\"', '"').replace("\\'", "'") for p in parts)


def _core_methods():
    src = _CORE.read_text()
    out = {}
    for m in _METHOD.finditer(src):
        name, _, body = m.groups()
        o = _OUT.search(body)
        rc3 = _RC3.search(body)
        guard = _GUARD.search(body)
        out[name] = dict(
            dims=tuple(d.strip() for d in o.group(1).split(",")) if o else None,
            rc3=_cjoin(rc3.group(1)) if rc3 else None,
            guard=_cjoin(guard.group(1)) if guard else None,
        )
    return out


def test_py_fields_match_core():
    from grid_codegen.abi_specs import ABI_SPECS

    methods = _core_methods()
    assert len(methods) > 60, "method parse broke (found too few pybind methods)"
    problems = []
    for key, spec in ABI_SPECS.items():
        base = methods.get(key)
        twin = methods.get(key + "_mujoco")
        if base is None and twin is None:
            if spec.py_out_dims or spec.py_rc3_msg or spec.py_twin_guard:
                problems.append(f"{key}: py_* fields set but no _core method exists")
            continue
        if base is not None:
            if spec.py_out_dims != base["dims"]:
                problems.append(f"{key}: py_out_dims {spec.py_out_dims!r} != core {base['dims']!r}")
            if spec.py_rc3_msg != base["rc3"]:
                problems.append(f"{key}: py_rc3_msg mismatch\n  spec: {spec.py_rc3_msg!r}\n  core: {base['rc3']!r}")
        if (twin["guard"] if twin else None) != spec.py_twin_guard:
            problems.append(f"{key}: py_twin_guard mismatch\n  spec: {spec.py_twin_guard!r}\n"
                            f"  core: {(twin['guard'] if twin else None)!r}")
    assert not problems, "\n".join(problems)


def test_every_specced_method_carries_out_dims():
    """Any spec row whose pybind method allocates an out array must carry
    py_out_dims (the C4 generator + the jax/torch reshape collapse key on it)."""
    from grid_codegen.abi_specs import ABI_SPECS

    methods = _core_methods()
    missing = [k for k, s in ABI_SPECS.items()
               if methods.get(k, {}).get("dims") and not s.py_out_dims]
    assert not missing, f"rows missing py_out_dims: {missing}"

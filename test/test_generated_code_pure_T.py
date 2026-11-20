"""CPU-only guard: generated device code computes in T.

A C++ decimal literal is a double and promotes any T operand it meets, and `pow` with an
integer exponent resolves to the double routine. Symbolic expressions are therefore
printed by `grid_codegen.helpers._expr_printer.expr_T` only, and this test scans whole
generated headers so that no other emitter reintroduces either form.

The second derivatives of SE(3) integration are written in double on purpose (their
coefficients cancel badly in single precision); they are the only exemption and are
listed by name.
"""
from __future__ import annotations

import contextlib
import os
import re

import pytest
import sympy as sp

from grid_assets import robot_urdf
from external.URDFParser.URDFParser import URDFParser
from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
from grid_codegen.helpers._expr_printer import assert_pure_T, expr_T, promoting_literals

DELIBERATE_DOUBLE = ("se3_d2_", "se3_retract_hessian", "integrator_hessian_device")
_FUNCTION = re.compile(r"^    (?:[\w:<>,\s\*&]*\s)?(\w+)\s*\(.*\)\s*(?:const\s*)?\{\s*$")
_TYPED = re.compile(r"(?:static_cast<T>|T)\(\s*-?[\d.eE+\-]+(?:\s*/\s*[\d.eE+\-]+)?\s*\)|\(T\)\s*-?[\d.]+(?:[eE][-+]?\d+)?")
_CONSTANT_STORE = re.compile(r"=\s*-?[\d.]+(?:[eE][-+]?\d+)?\s*;")
_NUMBER = r"(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
_OPERATOR = r"(?:[*/+\-]|[<>]=?|[=!]=)"
# a double literal as an operand next to a variable, an element or a parenthesised term
_MIXED = re.compile(r"(?:(?:[A-Za-z_]\w*|\]|\))\s*" + _OPERATOR + r"\s*" + _NUMBER + r")|(?:(?<![\w.])" + _NUMBER
                    + r"\s*" + _OPERATOR + r"\s*[A-Za-z_(])")
_INTEGER_POW = re.compile(r"(?<![\w.])pow\(\s*[^,()]+(?:\[[^\]]*\])?\s*,\s*\d+\s*\)")


def device_functions(source):
    """(name, first line, body lines) of every __device__ / __global__ function."""
    lines, out, qualifier, current, depth = source.splitlines(), [], "", None, 0
    for number, raw in enumerate(lines, 1):
        line = raw.split("//")[0]
        if current is None:
            stripped = raw.strip()
            if stripped.startswith(("__device__", "__global__", "__host__")):
                qualifier = "device" if ("__device__" in stripped or "__global__" in stripped) else "host"
            match = _FUNCTION.match(raw)
            if match and match.group(1) not in ("if", "for", "while", "switch"):
                current, depth = (match.group(1), number, [], qualifier), 1
                qualifier = ""
            continue
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            if current[3] == "device":
                out.append(current[:3])
            current = None
        else:
            current[2].append((number, line))
    return out


def impure_lines(source):
    found = []
    for name, _, body in device_functions(source):
        if name.startswith(DELIBERATE_DOUBLE):
            continue
        for number, line in body:
            text = _CONSTANT_STORE.sub(";", _TYPED.sub("K", line))
            if _MIXED.search(text) or _INTEGER_POW.search(line) or re.search(r"\bdouble\b", line):
                found.append((name, number, line.strip()[:160]))
    return found


def test_printer_types_every_constant():
    x, y, t = sp.symbols("a b theta")
    assert expr_T(2.0 * x * y / (x**2 + y**2)) == \
        "static_cast<T>(2.0)*a*b/(((a)*(a)) + ((b)*(b)))"
    assert expr_T(x / 2) == "(static_cast<T>(1.0/2.0))*a"
    assert expr_T(1 / x) == "(static_cast<T>(1)/((a)))"
    assert expr_T(0.0465 * sp.sin(t)) == "static_cast<T>(0.0465)*sin(theta)"
    assert expr_T(sp.Float(0.5)) == "0.5" and expr_T(sp.Integer(3)) == "3"
    for expression in (x**3 - 0.1 * y, sp.sqrt(x) / 3, x**sp.Rational(3, 2), -sp.cos(t) * 1.5):
        assert not promoting_literals(expr_T(expression))


def test_impure_text_is_rejected():
    for text in ("2.0*s_q[3]", "pow(s_q[4], 2)", "s_q[0]/3.0"):
        with pytest.raises(ValueError):
            assert_pure_T(text)
    assert impure_lines("    __device__\n    void f(T *a) {\n        a[0] = 2.0*a[1];\n    }\n") != []
    assert impure_lines("    __device__\n    void f(T *a) {\n        a[0] = static_cast<T>(2.0)*a[1]; a[1] = 0.0;\n    }\n") == []


@pytest.mark.parametrize("robot_id,floating", [("iiwa14", False), ("go2", True), ("g1", True)])
def test_generated_device_code_is_pure_T(robot_id, floating, tmp_path):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(robot_urdf(robot_id)), floating_base=floating)
        header = tmp_path / "grid.cuh"
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(output_path=str(header))
    found = impure_lines(header.read_text())
    assert not found, f"{len(found)} device lines compute outside T, first: {found[:5]}"

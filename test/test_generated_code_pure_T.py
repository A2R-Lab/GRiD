"""CPU-only regression check: generated device code computes in T.

A C++ floating literal without a suffix is a double and promotes any T operand it meets,
and `pow` with an integer exponent resolves to the double routine. Symbolic expressions are
therefore printed by `grid_codegen.helpers._expr_printer.expr_T` only, and this test scans
whole generated headers so that no other emitter reintroduces either form.

It is a scan of the text for known forms, not a proof about the arithmetic: a promotion
that needs type information to see (a double returned by a called function, say) passes.
The canaries below state what it does catch.

The second derivatives of SE(3) integration are written in double on purpose (their
coefficients cancel badly in single precision). They live in GLASS, are the only
exemption and are listed by exact name.
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

# Exact names, not prefixes: a new function is not exempt because of how it is called.
DELIBERATE_DOUBLE = frozenset({"se3_d2_coefs", "se3_retract_hessian"})
# Everything after the vendored GLASS section is written by GRiD. There the word `double`
# may appear only in the statements below: the constant model records, whose values are
# cast to T before any arithmetic, and the host timing printout. A statement is compared
# whole, after comments and string contents are removed.
GLASS_END = "} // namespace glass"
RECORD_STATEMENTS = frozenset({"double multiplier, offset, axis[3], pitch, T_parent_joint_zero[16]",
                               "double position_limit[2], velocity_limit, effort_limit",
                               "double T_body_frame[16]"})
HOST_TIMING = re.compile(r'printf\(""\s*,\s*time_delta_us_timespec\(start,end\)/static_cast<double>\(num_timesteps\)\)')
_FUNCTION = re.compile(r"^    (?:[\w:<>,\s\*&]*\s)?(\w+)\s*\(.*\)\s*(?:const\s*)?\{\s*$")
# A floating literal without a float suffix is a double: 1.5, 1., .5, 1e-5, 1.5e3.
_LITERAL = re.compile(r"(?<![\w.])(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")
_NUM = r"-?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
# A literal is typed when a cast to T encloses it and nothing else.
_TYPED = re.compile(r"(?:static_cast<T>|\bT)\(\s*" + _NUM + r"(?:\s*/\s*" + _NUM + r")?\s*\)|\(T\)\s*" + _NUM)
# Storing a literal converts it and computes nothing: `x = 0.0;`, `return 0.0;`, a braced list of constants.
_CONSTANT_STORE = re.compile(r"(?:(?<![=!<>+\-*/])=|\breturn)\s*" + _NUM + r"\s*(?=;)")
# A braced list holds constants only (numbers, names such as enumerators, nested lists): no operator.
_ITEM = r"(?:" + _NUM + r"|[A-Za-z_][\w:]*|\{\})"
_CONSTANT_LIST = re.compile(r"\{\s*(?:" + _ITEM + r"(?:\s*,\s*" + _ITEM + r")*)?\s*,?\s*\}")
_POW = re.compile(r"(?<![\w.])(?:std::)?pow\s*\(")


def code_of(raw):
    """The line without string contents and without its comment."""
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', raw).split("//")[0]


def integer_powers(line):
    """Every `pow(x, n)` with an integer literal n, whatever x contains."""
    found = []
    for match in _POW.finditer(line):
        depth, split, end = 1, None, None
        for at in range(match.end(), len(line)):
            c = line[at]
            depth += (c in "([") - (c in ")]")
            if c == "," and depth == 1:
                split = at
            if depth == 0:
                end = at
                break
        if split is not None and end is not None and re.fullmatch(r"\s*-?\d+\s*", line[split + 1:end]):
            found.append(line[match.start():end + 1])
    return found


def computes_outside_T(line):
    """True for code (comments and strings already removed) with a double literal that is
    neither cast to T nor merely stored, an integer `pow`, or the word `double`."""
    text, before = _CONSTANT_STORE.sub("", _TYPED.sub("K", line)), None
    while text != before:                                  # innermost lists first
        before, text = text, _CONSTANT_LIST.sub("{}", text)
    return bool(_LITERAL.search(text) or integer_powers(line) or re.search(r"\bdouble\b", line))


def device_functions(source):
    """(name, first line, body lines) of every __device__ / __global__ function."""
    lines, out, qualifier, current, depth = source.splitlines(), [], "", None, 0
    for number, raw in enumerate(lines, 1):
        line = code_of(raw)
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
        if name in DELIBERATE_DOUBLE:
            continue
        for number, line in body:
            if computes_outside_T(line):
                found.append((name, number, line.strip()[:160]))
    return found


def stray_double_lines(source):
    """Lines GRiD wrote, at any indentation and in any function, that compute outside T."""
    found, in_block_comment = [], False
    for raw in source[source.index(GLASS_END):].splitlines():
        line = code_of(raw)
        if in_block_comment or line.lstrip().startswith(("/*", "*")):
            in_block_comment = "*/" not in line and (in_block_comment or line.lstrip().startswith("/*"))
            continue
        kept = ";".join(statement for statement in line.split(";")
                        if statement.strip() not in RECORD_STATEMENTS and not HOST_TIMING.fullmatch(statement.strip()))
        if computes_outside_T(kept):
            found.append(line.strip()[:160])
    return found


def test_printer_types_every_constant():
    x, y, t = sp.symbols("a b theta")
    assert expr_T(2.0 * x * y / (x**2 + y**2)) == \
        "static_cast<T>(2.0)*a*b/(fma(a, a, b*b))"
    assert expr_T(x / 2) == "(static_cast<T>(1.0/2.0))*a"
    assert expr_T(1 / x) == "(static_cast<T>(1)/((a)))"
    assert expr_T(0.0465 * sp.sin(t)) == "static_cast<T>(0.0465)*sin(theta)"
    assert expr_T(sp.Float(0.5)) == "0.5" and expr_T(sp.Integer(3)) == "3"
    for expression in (x**3 - 0.1 * y, sp.sqrt(x) / 3, x**sp.Rational(3, 2), -sp.cos(t) * 1.5):
        assert not promoting_literals(expr_T(expression))


def test_printer_fuses_every_sum_of_products_explicitly():
    """`a*b + c` left to the compiler is fused at some call sites and not at others, which made
    two resource tiers of one kernel disagree in the last bit. The printer leaves no such sum."""
    a, b, c, d = sp.symbols("a b c d")
    assert expr_T(0.5 * a + 0.25 * b) == "fma(static_cast<T>(0.5), a, static_cast<T>(0.25)*b)"
    assert expr_T(a * b + c * d + 2) == "fma(a, b, fma(c, d, static_cast<T>(2)))"
    assert expr_T(a - b * c) == "fma(-b, c, a)"
    assert expr_T(a * b + 1) == "fma(a, b, static_cast<T>(1))"
    assert expr_T(a**2 + b) == "fma(a, a, b)"
    # Sums without a product, and divisions, are printed as written.
    assert expr_T(-a - 0.5) == "-a + static_cast<T>(-0.5)"
    assert expr_T(0.5 / a + b * c) == "fma(b, c, static_cast<T>(0.5)/a)"
    assert expr_T(-a / b + c * d) == "fma(c, d, -a/b)"
    for expression in (a * b + c * d - 0.5 * a + 2, 1 - 2 * (a**2 + b**2), a * (b + c * d) + d):
        assert not _unfused_sums(expr_T(expression)), expr_T(expression)
    assert _unfused_sums("a*b + c") and _unfused_sums("fma(a, b, c*d + e)") and not _unfused_sums("fma(a, b, c) + d")


def _unfused_sums(text):
    """Parenthesised groups (and the whole text) that both multiply and add at their own level."""
    found, stack = [], [[]]
    for ch in text:
        if ch == "(":
            stack.append([])
        elif ch == ")":
            group = "".join(stack.pop())
            if "*" in group and re.search(r"(?<![(,*/eE\s])\s*[+-]", group.strip()):
                found.append(group)
            stack[-1].append("G")
        elif ch == ",":
            group = "".join(stack[-1])
            if "*" in group and re.search(r"(?<![(,*/eE\s])\s*[+-]", group.strip()):
                found.append(group)
            stack[-1] = []
        else:
            stack[-1].append(ch)
    group = "".join(stack[0])
    if "*" in group and re.search(r"(?<![(,*/eE\s])\s*[+-]", group.strip()):
        found.append(group)
    return found


def _device(body):
    return f"    __device__\n    void f(T *a) {{\n        {body}\n    }}\n"


def _stray(body):
    return GLASS_END + f"\ntemplate<class T> __device__ inline void f(\n    const T *a, T *e) {{\n    {body}\n}}\n"


IMPURE = ("a[0] = 2.0*a[1];", "a[0] = a[1]/3.0;", "a[0] = 1e-5*a[1];", "a[0] = 1.*a[1];", "a[0] = .5*a[1];",
          "a[0] = fma(a[1],a[2],1.0);", "a[0] = (a[1] > 0.0) ? a[1] : a[2];", "a[0] = pow(a[1],2);",
          "a[0] = pow(sin(a[1]),2);", "a[0] = std::pow(a[1] + f(a[2], 3), 2);", "a[0] = T(2.0)*a[1] + 2.0;",
          "double b = a[0];", "a[0] = static_cast<double>(a[1]);", "T b[2] = {0.5*a[0], 1.0};",
          "out = record{1, {2.0*a[0]}};")
PURE = ("a[0] = static_cast<T>(2.0)*a[1]; a[1] = 0.0;", "a[0] = T(1e-5)*a[1];", "a[0] = (T)0.5*a[1];",
        "a[0] = pow(a[1], a[2]);", "a[0] = static_cast<T>(1.0/3.0)*a[1];", "return 0.0;", "T b[2] = {0.0, 1.0};",
        "out = joint_record{0,-1,joint_kind::floating,true,1.0,{0.0,1e-3,0.0},{1.0,0.0}};",
        "a[0] = 2*a[1]; // 2.0 in a comment, and double too", 'printf("%f 2.0 double", a[0]);', "a[0] = 1.5f*a[1];",
        "a[0] = x1e5*a[1];")


@pytest.mark.parametrize("body", IMPURE)
def test_canary_is_caught_by_both_scanners(body):
    assert impure_lines(_device(body)) != []
    assert stray_double_lines(_stray(body)) != []


@pytest.mark.parametrize("body", PURE)
def test_pure_line_passes_both_scanners(body):
    assert impure_lines(_device(body)) == []
    assert stray_double_lines(_stray(body)) == []


def test_impure_text_is_rejected_by_the_printer_check():
    for text in ("2.0*s_q[3]", "pow(s_q[4], 2)", "s_q[0]/3.0"):
        with pytest.raises(ValueError):
            assert_pure_T(text)


def test_exemptions_are_whole_statements_and_exact_names():
    # a function at column zero with a signature over two lines, as the model primitives are written
    stray = GLASS_END + "\ntemplate<class T> __device__ inline void f(\n    const T *a, double *e) {\n    double b[3]; e[0]=double(a[0])*2.;\n}\n"
    assert len(stray_double_lines(stray)) == 2
    record = "    int id, attached_body, joint_id; frame_kind kind; double T_body_frame[16];"
    timing = '    printf("Single Call ABA %fus\\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));'
    assert stray_double_lines(GLASS_END + "\n" + record + "\n" + timing + "\n") == []
    # the name of an exemption in a comment, in a string or next to other code exempts nothing
    for contaminated in ("double x=a[0]; // time_delta_us_timespec", "double x=a[0]; // double T_body_frame[16];",
                         'double x=a[0]; printf("time_delta_us_timespec");', "double T_body_frame[16]; double x=a[0];",
                         "double y = time_delta_us_timespec(start,end)*a[0];"):
        assert stray_double_lines(GLASS_END + "\n    " + contaminated + "\n") != [], contaminated
    # a block comment is not code
    assert stray_double_lines(GLASS_END + "\n/* double here\n * and 2.0*x here\n */\n") == []
    # only the named functions are exempt, not every function that starts like one
    named = "    __device__\n    void {}(T *a) {{\n        double b = 2.0*a[1];\n    }}\n"
    assert impure_lines(named.format("se3_d2_coefs")) == []
    assert impure_lines(named.format("se3_d2_coefs_new")) != []
    assert impure_lines(named.format("se3_retract_hessian_fast")) != []


# what the generator supports on a spherical joint
SPHERICAL_ALGORITHMS = ["inverse_dynamics", "crba", "minv", "forward_dynamics", "aba", "end_effector_pose",
                        "end_effector_pose_gradient", "end_effector_pose_hessian", "integrator", "integrator_gradient",
                        "inverse_dynamics_gradient", "forward_dynamics_gradient", "idsva_so_world_frame", "fdsva_so"]


def _urdf(robot_id):
    if robot_id == "spherical_arm":
        import external.URDFParser as parser_package
        return os.path.join(os.path.dirname(parser_package.__file__), "tests/fixtures/spherical_arm.urdf")
    return str(robot_urdf(robot_id))


@pytest.mark.parametrize("robot_id,floating", [("iiwa14", False), ("go2", True), ("g1", True), ("spherical_arm", False)])
def test_generated_device_code_is_pure_T(robot_id, floating, tmp_path):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(_urdf(robot_id), floating_base=floating)
        header = tmp_path / "grid.cuh"
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            output_path=str(header), algorithm_list=SPHERICAL_ALGORITHMS if robot_id == "spherical_arm" else None)
    source = header.read_text()
    found = impure_lines(source)
    assert not found, f"{len(found)} device lines compute outside T, first: {found[:5]}"
    stray = stray_double_lines(source)
    assert not stray, f"{len(stray)} lines after the vendored GLASS section compute outside T, first: {stray[:5]}"

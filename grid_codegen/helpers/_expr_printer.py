"""The one printer that turns a symbolic expression into device code.

Every generated function is templated on T and computes in T. A C++ decimal literal is a
double, and a double next to a T operand promotes the whole expression, so symbolic
expressions are never pasted through `str()` or `sp.ccode()`: they go through `expr_T`,
which prints

  - every non-integer number as `static_cast<T>(...)`,
  - integer powers as products (`x*x`, never `pow(x, 2)`),
  - reciprocals and other powers with their constants in T.

`assert_pure_T` is the check behind it and is also used by the tests on generated headers.
"""
import re
import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import PRECEDENCE

_LITERAL = re.compile(r"(?<![\w.\]])(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
_TYPED = re.compile(r"static_cast<T>\(\s*-?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*\d+\.\d*)?\s*\)")
_MAX_PRODUCT_POWER = 4


def promoting_literals(text):
    """Decimal literals in `text` that are not the whole argument of a cast to T."""
    return _LITERAL.findall(_TYPED.sub("K", text))


def assert_pure_T(text):
    bad = promoting_literals(text)
    if bad or re.search(r"(?<![\w.])pow\(\s*[^,]+,\s*\d+\s*\)", text):
        raise ValueError(f"expression is not pure T (double literal or integer pow): {text[:200]}")
    return text


class _TPrinter(C99CodePrinter):
    def _print_Float(self, expr):
        return f"static_cast<T>({float(expr)!r})"

    def _print_Rational(self, expr):
        return f"static_cast<T>({int(expr.p)}.0/{int(expr.q)}.0)"

    def _print_Pow(self, expr):
        base, exp = expr.base, expr.exp
        b = self.parenthesize(base, PRECEDENCE["Atom"])
        if exp.is_Integer:
            n = int(exp)
            if 2 <= n <= _MAX_PRODUCT_POWER:
                return "(" + "*".join([b] * n) + ")"
            if -_MAX_PRODUCT_POWER <= n <= -1:
                return "(static_cast<T>(1)/(" + "*".join([b] * -n) + "))"
        if exp == sp.Rational(1, 2):
            return f"sqrt({self._print(base)})"
        if exp == sp.Rational(-1, 2):
            return f"(static_cast<T>(1)/sqrt({self._print(base)}))"
        e = self._print(exp)                  # a Float or Rational exponent is already typed
        return f"pow({self._print(base)}, {e if e.startswith('static_cast<T>') else 'static_cast<T>(' + e + ')'})"


_PRINTER = _TPrinter()

# The floating base quaternion. Its rotation is normalised by the squared norm, and the
# symbolic matrices repeat that division in every entry. `expr_T` prints each occurrence
# of the squared norm as one call of the header's `grid_quat_norm2`, the same function
# `grid_base_rotation` divides by, so the header has a single definition of how the base
# quaternion is normalised (and a single place where `assume_unit_quaternions` removes
# it). The division itself stays where the expression has it: `x/n` and `x*(1/n)` round
# differently, and the floating forward dynamics gradient in single precision is
# sensitive enough to that for an integrator cell to leave its measured bound.
_FB_QUAT = sp.symbols("q1_fb q2_fb q3_fb q4_fb")
_FB_NORM2 = sum(q**2 for q in _FB_QUAT)
_FB_NORM2_TEXT = _PRINTER.doprint(_FB_NORM2)
FB_NORM2_CALL = "grid_quat_norm2<T>(q1_fb, q2_fb, q3_fb, q4_fb)"


def _check_base_quaternion_divisors(value):
    for power in value.atoms(sp.Pow):
        if power.exp.is_negative and power.base.has(*_FB_QUAT) and power.base != _FB_NORM2:
            raise ValueError(f"base quaternion divided by something other than its squared norm: {power}")


def expr_T(value):
    """Device-code text of `value` (a sympy expression or a number), computed in T."""
    value = sp.sympify(value)
    if value.is_Integer:
        return str(int(value))
    if value.is_Number:
        return repr(float(value))            # the caller's cast to T makes it a T constant
    text = assert_pure_T(_PRINTER.doprint(value))
    if value.has(*_FB_QUAT):
        _check_base_quaternion_divisors(value)
        text = text.replace(_FB_NORM2_TEXT, FB_NORM2_CALL)
    return text

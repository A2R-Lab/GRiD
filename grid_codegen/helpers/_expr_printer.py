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

    def _print_Add(self, expr):
        """Sums of products as nested `fma`, with a fixed order.

        `a*b + c` in T may or may not be contracted into a fused multiply-add, and the
        compiler decides that per call site: the same helper inlined next to shared memory
        and next to global memory was contracted differently, so two resource tiers of one
        kernel stopped agreeing bit for bit. Writing the fused form explicitly removes the
        choice. Terms that are not products are summed first, then each product is folded
        in with one `fma`.
        """
        for quat, norm2 in zip(_QUATERNIONS, _NORM2):
            if set(norm2.args) <= set(expr.args):     # the squared norm is one call, not four products
                call = sp.Symbol("grid_quat_norm2<T>(" + ", ".join(q.name for q in quat) + ")")
                return self._print(sp.Add(*(set(expr.args) - set(norm2.args)), call, evaluate=False)) \
                    if len(expr.args) > 4 else call.name
        plain, products = [], []
        for term in expr.as_ordered_terms():
            split = self._product_split(term)
            (products if split else plain).append(split or term)
        if not products:
            return super()._print_Add(expr)
        if plain:
            acc = super()._print_Add(sp.Add(*plain)) if len(plain) > 1 else self._fma_arg(plain[0])
        else:
            a, b = products.pop()
            acc = f"{self.parenthesize(a, PRECEDENCE['Mul'])}*{self.parenthesize(b, PRECEDENCE['Mul'])}"
        for a, b in reversed(products):
            acc = f"fma({self._fma_arg(a)}, {self._fma_arg(b)}, {acc})"
        return acc

    def _fma_arg(self, value):
        """An integer argument would make the call ambiguous between the T and double overloads."""
        return f"static_cast<T>({int(value)})" if value.is_Integer else self._print(value)

    @staticmethod
    def _product_split(term):
        """(a, b) with term == a*b when `term` multiplies two or more factors, else None.

        A division stays inside `b` as the expression has it (see the note on quaternions below)."""
        if term.is_Pow and term.exp.is_Integer and 2 <= int(term.exp) <= _MAX_PRODUCT_POWER:
            return term.base, term.base ** (int(term.exp) - 1)
        if not term.is_Mul:
            return None
        coeff, rest = term.as_coeff_Mul()
        numerator, denominator = sp.fraction(rest)
        if numerator == 1:
            return None                       # c/x: a division, which is never fused
        if coeff not in (1, -1):
            return coeff, rest
        factors = numerator.as_ordered_factors()
        if len(factors) < 2:
            return None                       # x/y or -x/y: likewise
        return coeff * factors[0], sp.Mul(*factors[1:]) / denominator

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

# Quaternions: the floating base (q*_fb) and every spherical joint (q*_sph). Their
# rotations are normalised by the squared norm, and the symbolic matrices repeat that
# division in every entry. `expr_T` prints each occurrence of the squared norm as one call
# of the header's `grid_quat_norm2`, the same function `grid_base_rotation` divides by, so
# the header has a single definition of how a quaternion is normalised (and a single place
# where `assume_unit_quaternions` removes it). The division itself stays where the
# expression has it: `x/n` and `x*(1/n)` round differently, and the floating forward
# dynamics gradient in single precision is sensitive enough to that for an integrator cell
# to leave its measured bound.
_QUATERNIONS = tuple(sp.symbols(" ".join(f"q{k}_{tag}" for k in (1, 2, 3, 4))) for tag in ("fb", "sph"))
_NORM2 = tuple(sum(q**2 for q in quat) for quat in _QUATERNIONS)
_NORM2_CALLS = tuple((_PRINTER.doprint(norm2), "grid_quat_norm2<T>(" + ", ".join(q.name for q in quat) + ")")
                     for quat, norm2 in zip(_QUATERNIONS, _NORM2))
FB_NORM2_CALL = _NORM2_CALLS[0][1]


def _check_quaternion_divisors(value):
    for power in value.atoms(sp.Pow):
        if power.exp.is_negative and any(power.base.has(*quat) and power.base != norm2
                                         for quat, norm2 in zip(_QUATERNIONS, _NORM2)):
            raise ValueError(f"quaternion divided by something other than its squared norm: {power}")


def expr_T(value):
    """Device-code text of `value` (a sympy expression or a number), computed in T."""
    value = sp.sympify(value)
    if value.is_Integer:
        return str(int(value))
    if value.is_Number:
        return repr(float(value))            # the caller's cast to T makes it a T constant
    text = assert_pure_T(_PRINTER.doprint(value))
    if any(value.has(*quat) for quat in _QUATERNIONS):
        _check_quaternion_divisors(value)
        for norm2_text, call in _NORM2_CALLS:
            text = text.replace(norm2_text, call)
    return text

"""Per-algorithm suggested thread counts, derived from the emitted header text.

A consumer that fuses several GRiD algorithms into one kernel (GATO's merit
kernel runs one forward dynamics and one tracking cost per block) needs a thread
count that covers the widest block-parallel loop of the algorithms it calls, not
the header-wide MAX_PERF_LEVEL_THREADS. The generator knows every loop bound it
emits, so this module reads them back from the finished header: it indexes every
function body, follows the calls from each algorithm's kernels through the
device and inner layers, and takes the widest ``for(int i = threadIdx.x + ...;
i < BOUND; ...)`` bound on that call tree. The suggestion is that width rounded
up to a multiple of 32 and clamped to [32, 1024]. It is GPU-independent and
untuned; ``launch_cfg<GRID_ALGO_*>::THREADS`` stays the tuned override.

The emitted block reopens the ``grid`` namespace at the end of the header:

    const int FORWARD_DYNAMICS_SUGGESTED_THREADS = 64;
    const int TRACKING_COST_SUGGESTED_THREADS = 32;

Bounds that are not integer literals (a handful of per-body counts such as
``vc_count``, at most 6) are skipped and reported by ``derive`` so a test can
keep the list of accepted names explicit.
"""
from __future__ import annotations

import re
from collections import defaultdict

from grid_codegen.kernel_attrs import KERNEL_OVERLOADS

_LOOP_RE = re.compile(
    r"for\s*\(int\s+(\w+)\s*=\s*threadIdx\.x(?:\s*\+\s*threadIdx\.y\s*\*\s*blockDim\.x)?\s*;"
    r"\s*\1\s*<\s*([^;]+?)\s*;")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;(){}]*?>)?\s*\(")
_DEF_RE = re.compile(r"^\s*(?!(?:for|if|else|while|switch|do|return|namespace|struct|class|extern|template)\b)"
                     r"[\w:<>,\*&\s]+?\s\**(\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{\s*$")
_LOCAL_CONST_RE = re.compile(r"\b(?:const\s+|constexpr\s+|static\s+)*int\s+(\w+)\s*=\s*([^;,{}]+?)\s*;")
_HEADER_CONST_RE = re.compile(r"^\s*(?:static\s+)?(?:const|constexpr) int (\w+) = (\d+);\s*$", re.M)
_DEFINE_RE = re.compile(r"^\s*#define (\w+) (\d+)\s*$", re.M)
_CONST_ARRAY_RE = re.compile(r"\b(?:static\s+)?constexpr int (\w+)\[\] = \{([^}]*)\}")
_ARRAY_READ_RE = re.compile(r"\b(\w+)\[[^\[\]]*\]")
_INDEX_RE = re.compile(r"\[[^\[\]]*\]")
_ARITH_RE = re.compile(r"[+\-*/%]|\b(?:sin|cos|tanh|sqrt|exp|log|dot_prod|fma)\b")
_CONTROL = {"for", "if", "while", "switch", "sizeof", "static_cast", "reinterpret_cast", "return"}

MIN_THREADS = 32
MAX_THREADS = 1024


_TEMPLATE_DEFAULT_RE = re.compile(r"\bint\s+([A-Za-z_]\w*)\s*=\s*(\d+)")


def _strip_comment(line: str) -> str:
    i = line.find("//")
    return line if i < 0 else line[:i]


def index_functions(text: str) -> dict[str, list[str]]:
    """Map every function name to its bodies (overloads and template variants
    share a name). A definition is a line that ends with ``{`` and looks like a
    function head; the body runs until the brace depth returns to zero."""
    bodies: dict[str, list[str]] = defaultdict(list)
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = _strip_comment(lines[i])
        m = _DEF_RE.match(raw)
        if m is None or m.group(1) in _CONTROL:
            i += 1
            continue
        name = m.group(1)
        depth = raw.count("{") - raw.count("}")
        j = i + 1
        chunk = [raw]
        # keep the template line in front of the body: a loop bound may be a
        # template parameter whose default is the width to read
        k = i - 1
        while k >= 0 and k >= i - 3 and _strip_comment(lines[k]).strip() in ("__device__", "__host__", "__host__ __device__", "__device__ __forceinline__", "__forceinline__"):
            k -= 1
        if k >= 0 and _strip_comment(lines[k]).lstrip().startswith("template"):
            chunk.insert(0, _strip_comment(lines[k]))
        while j < n and depth > 0:
            l = _strip_comment(lines[j])
            depth += l.count("{") - l.count("}")
            chunk.append(l)
            j += 1
        bodies[name].append("\n".join(chunk))
        i = j if j > i else i + 1
    return bodies


def _eval_bound(expr: str, consts: dict[str, int]) -> int | None:
    expr = expr.strip()
    if re.fullmatch(r"\d+", expr):
        return int(expr)
    sub = re.sub(r"\b[A-Za-z_]\w*\b", lambda mm: str(consts[mm.group(0)]) if mm.group(0) in consts else mm.group(0), expr)
    if re.fullmatch(r"[\d+\-*/() ]+", sub):
        try:
            return int(eval(sub, {"__builtins__": {}}, {}))  # arithmetic on integer literals only
        except Exception:
            return None
    return None


def _loop_body(body: str, start: int) -> str:
    """The statements inside the loop whose header starts at ``start`` (up to the
    matching brace; a braceless single statement up to its semicolon)."""
    close = body.find(")", start)
    k = close + 1
    while k < len(body) and body[k] in " \t":
        k += 1
    if k < len(body) and body[k] == "{":
        depth = 0
        for j in range(k, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    return body[k + 1:j]
        return body[k + 1:]
    end = body.find(";", k)
    return body[k:end + 1] if end >= 0 else body[k:]


def _strip_branches(text: str) -> str:
    """Remove `if (...)`, `if constexpr (...)`, `else`, `continue;` and braces so a
    guarded copy (`if (ind < n) a[ind] = b[ind]; else a[ind] = 0;`) reads as its
    assignments alone. The condition is dropped with balanced parentheses."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        m = re.match(r"\bif\s*(constexpr\s*)?\(", text[i:])
        if m:
            j = i + m.end() - 1
            depth = 0
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    flat = "".join(out)
    flat = re.sub(r"\belse\b|\bcontinue\s*;|[{}]", " ", flat)
    return flat


def is_copy_loop(statements: str) -> bool:
    """True when the loop only moves or zeroes values: every statement is an
    assignment whose right-hand side carries no arithmetic once the index
    expressions are removed (``a[i] = b[j]``, ``a[i] = static_cast<T>(0)``)."""
    stripped = _strip_branches(statements.replace("->", "."))
    stripped = _INDEX_RE.sub("[]", stripped)
    stripped = _INDEX_RE.sub("[]", stripped)   # nested index expressions
    stripped = re.sub(r"static_cast<\w+>", "", stripped)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    stmts = [t.strip() for t in stripped.split(";") if t.strip()]
    if not stmts:
        return True
    for st in stmts:
        if re.search(r"\bfor\b|\bwhile\b|\?", st):
            return False
        if "+=" in st or "-=" in st or "*=" in st or "/=" in st:
            return False
        if st.count("=") != 1:
            return False
        rhs = st.split("=", 1)[1]
        if _ARITH_RE.search(rhs):
            return False
    return True


def body_loop_widths(body: str, consts: dict[str, int], compute_only: bool = True) -> tuple[int, set[str]]:
    """Widest loop bound in one body and the bounds that could not be read.
    With ``compute_only`` the copy and zero-fill loops are left out: a copy
    striding over a few more elements per thread costs nothing against the
    serial compute of the loops that do arithmetic."""
    local = dict(consts)
    # a template parameter with an integer default (`int COUNT = 12`) reads as
    # its default: the widest instantiation the header itself emits
    head = body.split("\n", 1)[0]
    if head.lstrip().startswith("template"):
        for mm in _TEMPLATE_DEFAULT_RE.finditer(head):
            local.setdefault(mm.group(1), int(mm.group(2)))
    # a constexpr table read `t[i]` is bounded by the table's largest entry
    arrays = {mm.group(1): max(int(x) for x in re.findall(r"-?\d+", mm.group(2)) or [0])
              for mm in _CONST_ARRAY_RE.finditer(body)}
    for mm in _LOCAL_CONST_RE.finditer(body):
        rhs = _ARRAY_READ_RE.sub(lambda a: str(arrays[a.group(1)]) if a.group(1) in arrays else a.group(0), mm.group(2))
        val = _eval_bound(rhs, local)
        if val is not None:
            local.setdefault(mm.group(1), val)
    widest = 0
    unknown: set[str] = set()
    for mm in _LOOP_RE.finditer(body):
        if compute_only and is_copy_loop(_loop_body(body, mm.start())):
            continue
        val = _eval_bound(mm.group(2), local)
        if val is None:
            unknown.add(mm.group(2).strip())
        else:
            widest = max(widest, val)
    return widest, unknown


def call_tree_width(roots, bodies: dict[str, list[str]], consts: dict[str, int],
                    compute_only: bool = True) -> tuple[int, set[str], bool]:
    """Widest loop reachable from ``roots`` through the indexed functions.
    Returns (width, unknown bounds, found) where found says a root existed."""
    seen: set[str] = set()
    stack = [r for r in roots if r in bodies]
    found = bool(stack)
    widest = 0
    unknown: set[str] = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for body in bodies[name]:
            w, u = body_loop_widths(body, consts, compute_only)
            widest = max(widest, w)
            unknown |= u
            for cm in _CALL_RE.finditer(body):
                callee = cm.group(1)
                if callee in bodies and callee not in seen:
                    stack.append(callee)
    return widest, unknown, found


def _round_up(width: int, fallback: int) -> int:
    if width <= 0:
        return fallback
    return max(MIN_THREADS, min(MAX_THREADS, 32 * ((width + 31) // 32)))


def _kernel_roots(key: str) -> list[str]:
    roots = []
    for sym, _sig in KERNEL_OVERLOADS.get(key, []):
        base = sym.split("<")[0]
        if base not in roots:
            roots.append(base)
    if not roots:
        roots = [key + "_kernel"]
    return roots


TRACKING_COST_ROOTS = ("tracking_cost", "tracking_cost_gradient", "tracking_cost_hessian", "tracking_cost_fc")


def derive(text: str, keys, compute_only: bool = True) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Suggested thread count per algorithm key present in ``text`` (plus
    ``tracking_cost`` when the plant preset is emitted) and the unreadable loop
    bounds met on each call tree."""
    consts = {m.group(1): int(m.group(2)) for m in _HEADER_CONST_RE.finditer(text)}
    consts.update({m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(text)})
    fallback = consts.get("MAX_PERF_LEVEL_THREADS", MAX_THREADS)
    bodies = index_functions(text)
    values: dict[str, int] = {}
    unknown: dict[str, set[str]] = {}
    for key in keys:
        width, unk, found = call_tree_width(_kernel_roots(key), bodies, consts, compute_only)
        if not found:
            continue
        values[key] = _round_up(width, fallback)
        if unk:
            unknown[key] = unk
    width, unk, found = call_tree_width(TRACKING_COST_ROOTS, bodies, consts, compute_only)
    if found:
        values["tracking_cost"] = _round_up(width, fallback)
        if unk:
            unknown["tracking_cost"] = unk
    return values, unknown


def emit_block(namespace: str, values: dict[str, int]) -> list[str]:
    lines = [
        "",
        "// Suggested thread counts: per algorithm, the smallest multiple of 32 that covers the",
        "// widest block-parallel loop reachable from its kernel, read from the emitted code at",
        "// generation time (grid_codegen/suggested_threads.py). GPU-independent and untuned: a",
        "// consumer that fuses several algorithms into one kernel takes the max over the ones it",
        "// calls. launch_cfg<GRID_ALGO_*>::THREADS stays the tuned per-GPU override. Clamped to",
        "// [32, 1024]; algorithms whose loops exceed 1024 stride, so 1024 is the honest answer.",
        "namespace " + namespace + " {",
    ]
    for key, val in values.items():
        lines.append("    const int " + key.upper() + "_SUGGESTED_THREADS = " + str(val) + ";")
    lines.append("}")
    return lines

"""Host-launch thread-clamp post-pass: pure string transforms applied to the
finished generated source (no generator state). Split out of
GRiDCodeGenerator.py (H4 hygiene wave, 2026-08-27) verbatim.
"""
import re


_HOST_CLAMP_HELPER = """\
    // Clamp a requested launch thread count against the LAUNCHED kernel's own
    // cudaFuncAttributes cap (register pressure / __launch_bounds__). A request
    // above the cap is otherwise silently rejected at launch time: the stream
    // stays empty, sync succeeds, and the output buffer keeps stale contents.
    // Applied by codegen to every host-wrapper launch that takes thread_dimms.
    __host__ inline dim3 grid_host_clamp_threads(const void *kernel_fn, dim3 requested) {
        cudaFuncAttributes _attr;
        if (cudaFuncGetAttributes(&_attr, kernel_fn) != cudaSuccess) {
            cudaGetLastError();  // swallow -- fall back to the requested dims
            return requested;
        }
        unsigned _cap = (_attr.maxThreadsPerBlock > 0) ? (unsigned)_attr.maxThreadsPerBlock : requested.x;
        if (requested.x > _cap) requested.x = _cap;
        return requested;
    }
"""


def _extract_kernel_expr(before_launch):
    """Given the text of a line up to (not including) `<<<`, return the trailing
    kernel expression `name<targs>` (balanced <>), or None if unparseable."""
    s = before_launch.rstrip()
    if not s.endswith(">"):
        # untemplated kernel launch: trailing identifier
        m = re.search(r"([A-Za-z_]\w*)$", s)
        return m.group(1) if m else None
    depth = 0
    i = len(s) - 1
    while i >= 0:
        c = s[i]
        if c == ">":
            depth += 1
        elif c == "<":
            depth -= 1
            if depth == 0:
                break
        i -= 1
    if depth != 0 or i <= 0:
        return None
    m = re.search(r"([A-Za-z_]\w*)\s*$", s[:i])
    return (m.group(1) + s[i:]) if m else None


def _apply_host_thread_clamp_pass(code_str):
    lines = code_str.split("\n")
    # Overloaded kernels (same name, different parameter lists — e.g. the
    # qdd-flag variants) make `&name<targs>` ambiguous; detect them by counting
    # __global__ definitions per name and leave those launches unclamped.
    _kernel_defs = {}
    for _i, _ln in enumerate(lines):
        if "__global__" not in _ln or _ln.lstrip().startswith("//"):
            continue
        for _j in range(_i, min(_i + 4, len(lines))):  # __global__ / __launch_bounds__ / void name(
            _m = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\(", lines[_j])
            if _m:
                _kernel_defs[_m.group(1)] = _kernel_defs.get(_m.group(1), 0) + 1
                break
    _overloaded = {name for name, cnt in _kernel_defs.items() if cnt > 1}
    # inject the helper right after the grid_workspace_slot() block (early in
    # namespace grid, before every host wrapper)
    out = []
    injected = False
    for ln in lines:
        out.append(ln)
        if not injected and "return blockIdx.x + blockIdx.y*gridDim.x;" in ln:
            pass  # helper goes after the CLOSING brace, handled below
        if not injected and ln.strip() == "}" and len(out) >= 2 and \
                "return blockIdx.x + blockIdx.y*gridDim.x;" in out[-2]:
            out.append(_HOST_CLAMP_HELPER)
            injected = True
    if not injected:
        raise RuntimeError("host-thread-clamp pass: grid_workspace_slot anchor not found")
    lines = out

    out = []
    var_n = 0
    prev_expr = None
    prev_var = None
    rewritten = 0
    skipped = 0
    for ln in lines:
        if "<<<" in ln and "thread_dimms" in ln and not ln.lstrip().startswith("//"):
            expr = _extract_kernel_expr(ln.split("<<<")[0])
            if expr is None:
                warnings.warn(f"host-thread-clamp pass: unparseable launch line left unclamped: {ln.strip()[:100]}")
                out.append(ln)
                skipped += 1
                prev_expr = None
                continue
            if expr.split("<")[0] in _overloaded:
                # &name<targs> is ambiguous across parameter-list overloads
                out.append(ln)
                skipped += 1
                prev_expr = None
                continue
            stripped = ln.lstrip()
            indent = ln[:len(ln) - len(stripped)]
            if stripped.startswith("else") and expr == prev_expr and prev_var is not None:
                var = prev_var  # sibling branch launches the same instantiation
            else:
                var_n += 1
                var = f"_grid_thr_clamped_{var_n}"
                out.append(f"{indent}dim3 {var} = grid_host_clamp_threads((const void*)&{expr}, thread_dimms);")
            out.append(ln.replace("thread_dimms", var))
            rewritten += 1
            prev_expr, prev_var = expr, var
        else:
            if "<<<" not in ln:
                # only a directly-adjacent else-branch may reuse the clamp var
                if ln.strip() and not ln.lstrip().startswith("else"):
                    prev_expr = None
            out.append(ln)
    if rewritten == 0 and skipped == 0 and "thread_dimms" in code_str:
        # thread_dimms exists but no launch line matched: the emitter's launch shape
        # drifted out from under this pass. A header with NO thread_dimms at all is a
        # legitimate restricted-profile regen (no host wrappers emitted) — pass through.
        raise RuntimeError("host-thread-clamp pass: thread_dimms present but no launches matched (emitter drift?)")
    return "\n".join(out)

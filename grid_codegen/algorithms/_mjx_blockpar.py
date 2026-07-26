"""Shared block-parallel emit primitives for the mjx second-order epilogues.

The idsva_so and fdsva_so mjx epilogues both assemble their output slabs
BLOCK-PARALLEL: the whole block cooperates on one k at a time, spreading each
dense nv^2 op across all threads over block-shared scratch (no per-thread
register-array spill; see docs/agent_debugging_guide.md 1u). These are the loop
primitives both epilogues use so the "make this op block-strided" logic lives in
ONE place instead of being duplicated per algorithm file.
"""


def bpctrl(var, n):
    """Block-strided loop CONTROL (no trailing brace) over [0, n): each thread walks
    a disjoint stride of the index. Substitutes into both ``for(...) stmt;`` and
    ``for(...) {`` forms. The whole block executes it cooperatively (used INSIDE a
    plain ``for k`` so a slab's nv^2 work is spread across all threads)."""
    N = str(n)
    return ("for (int " + var + " = threadIdx.x + threadIdx.y*blockDim.x; " + var
            + " < " + N + "; " + var + " += blockDim.x*blockDim.y)")


def bpfor(var, n):
    """Block-strided loop header WITH the opening brace (``bpctrl`` + `` {``)."""
    return bpctrl(var, n) + " {"


def stride_rc(lines, n):
    """Make the full-nv ``for r``/``for c`` loops of a serial emit block block-strided
    (parallelize the op over rows / columns). Leaves ``for a<3`` and the ``for r/c=3``
    base-block copy loops untouched. Numerically identical to the serial form — only
    which thread computes each element changes; every (col,row) is written by exactly
    one thread. A ``__syncthreads()`` must separate a producer op from any consumer
    that reads an element a DIFFERENT thread wrote (the caller inserts those)."""
    N = str(n)
    reps = [("for (int r = 0; r < " + N + "; r++)", bpctrl("r", n)),
            ("for (int c = 0; c < " + N + "; c++)", bpctrl("c", n))]
    out = []
    for ln in lines:
        for old, new in reps:
            ln = ln.replace(old, new)
        out.append(ln)
    return out

"""Split wrapper_template.cu into the translation units of the multi-TU build.

The template stays ONE file (every drift test, hand-written body and doc tool
keeps working on it); fences — plain comment lines, invisible to the
single-unit compile — say which unit a block belongs to:

    // [[unit: <family>]]            ... // [[unit: end]]
    // [[unit: <family>; surface: jax]]   (the block lives inside the
                                          `#if defined(GRID_RBD_WITH_JAX)` section;
                                          the unit re-wraps it in that gate)
    // [[unit: prelude; surface: jax|torch]]  (static helpers / macros / includes
                                          a surface's blocks need: copied into EVERY
                                          family unit that has a block of that
                                          surface, in template order, before them)

Everything outside a fence is the CORE unit (contexts, lifecycle, metadata,
registration aggregator, torch library definition). Fences do not nest.

The generated regions carry per-key fences emitted by wrapper_body_gen /
wrapper_plant_gen (they know the key -> family map, abi_specs.FAMILY_OF); the
hand-written blocks carry hand-placed fences. `split()` is pure text: no
preprocessing, no key knowledge. `check_partition()` is the drift referee —
every template line must land in exactly one unit (core, one family, or one
prelude), and every fence must be well-formed.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

FENCE_RE = re.compile(r"^\s*//\s*\[\[unit:\s*([A-Za-z0-9_-]+)\s*(?:;\s*surface:\s*(jax|torch))?\s*\]\]\s*$")
END = "end"
PRELUDE = "prelude"
CORE = "core"
SURFACE_GATE = {"jax": "GRID_RBD_WITH_JAX", "torch": "GRID_RBD_WITH_TORCH"}


@dataclass
class Block:
    unit: str              # family name, "core", or "prelude"
    surface: str | None    # None | "jax" | "torch"
    start: int             # 1-based template line of the first body line
    lines: list[str] = field(default_factory=list)


class FenceError(ValueError):
    pass


def parse_blocks(text: str) -> list[Block]:
    """Template text -> ordered blocks. Core text between fences becomes core
    blocks (surface None). Fence lines themselves belong to no block."""
    blocks: list[Block] = []
    cur = Block(CORE, None, 1)
    open_fence_line = None
    for n, line in enumerate(text.split("\n"), 1):
        m = FENCE_RE.match(line)
        if not m:
            cur.lines.append(line)
            continue
        name, surface = m.group(1), m.group(2)
        if name == END:
            if open_fence_line is None:
                raise FenceError(f"line {n}: [[unit: end]] without an open fence")
            blocks.append(cur)
            cur = Block(CORE, None, n + 1)
            open_fence_line = None
        else:
            if open_fence_line is not None:
                raise FenceError(f"line {n}: fence opened inside the fence from line {open_fence_line} (no nesting)")
            blocks.append(cur)
            cur = Block(name, surface, n + 1)
            open_fence_line = n
    if open_fence_line is not None:
        raise FenceError(f"fence opened at line {open_fence_line} never closed")
    blocks.append(cur)
    return [b for b in blocks if b.lines]


def families_in(blocks: list[Block]) -> list[str]:
    seen: list[str] = []
    for b in blocks:
        if b.unit not in (CORE, PRELUDE) and b.unit not in seen:
            seen.append(b.unit)
    return seen


def _wrap(surface: str | None, body: str) -> str:
    if surface is None:
        return body
    gate = SURFACE_GATE[surface]
    return f"#if defined({gate})\n{body}\n#endif  // {gate} (unit surface)"


def unit_text(blocks: list[Block], unit: str, common_include: str = '#include "wrapper_common.cuh"') -> str:
    """The source of one unit. Core = every core block in order (the template's
    own prelude is a core block, so core needs no extra include). A family unit
    = the common include, then for each surface the family uses, that surface's
    prelude blocks, then the family's blocks, each surface-gated."""
    if unit == CORE:
        return "\n".join("\n".join(b.lines) for b in blocks if b.unit == CORE)
    mine = [b for b in blocks if b.unit == unit]
    if not mine:
        raise KeyError(unit)
    surfaces = []
    for b in mine:
        if b.surface not in surfaces:
            surfaces.append(b.surface)
    parts = [f"// wrapper unit `{unit}` — generated from wrapper_template.cu by grid_rbd._units (do not edit)",
             common_include]
    for surface in surfaces:
        if surface is not None:
            for p in blocks:
                if p.unit == PRELUDE and p.surface == surface:
                    parts.append(_wrap(surface, "\n".join(p.lines)))
        for b in mine:
            if b.surface == surface:
                parts.append(_wrap(surface, "\n".join(b.lines)))
    return "\n".join(parts) + "\n"


def split(text: str) -> dict[str, str]:
    """{unit name: source text} for core + every family present in the template."""
    blocks = parse_blocks(text)
    out = {CORE: unit_text(blocks, CORE)}
    for fam in families_in(blocks):
        out[fam] = unit_text(blocks, fam)
    return out


def check_partition(text: str) -> dict[str, int]:
    """Drift referee: parse (raising on malformed fences) and return the line
    count per unit; the sum over units + fence lines + prelude lines must equal
    the template's line count exactly (nothing dropped, nothing duplicated)."""
    lines = text.split("\n")
    blocks = parse_blocks(text)
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.unit if b.surface is None or b.unit != PRELUDE else f"{PRELUDE}:{b.surface}"] = \
            counts.get(b.unit if b.surface is None or b.unit != PRELUDE else f"{PRELUDE}:{b.surface}", 0) + len(b.lines)
    fences = sum(1 for l in lines if FENCE_RE.match(l))
    if sum(counts.values()) + fences != len(lines):
        raise FenceError(f"partition mismatch: {sum(counts.values())} block lines + {fences} fences != {len(lines)} template lines")
    for b in blocks:
        if b.unit == PRELUDE and b.surface is None:
            raise FenceError(f"prelude block at line {b.start} has no surface")
    return counts

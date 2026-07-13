"""The §1t catch-net: every kernel's launch-sizing macro must COVER what that kernel carves.

WHY THIS EXISTS (docs/agent_debugging_guide.md §1t). A kernel's shared arena is sized from ONE number
-- `<ALGO>_DYNAMIC_SHARED_MEM_BYTES<T,TIER>()`, which the host passes to the launch -- but the arena is
CARVED from a completely independent computation (the t_buffers list + temp_mem_size the emitter hands
`gen_declare_shared_arena`). Nothing tied the two together. When the fdsva_so `idsva_cold` spill rung
applied its saving to the TOTAL instead of inside the `max()`, the macro under-counted by 4308 bytes:
the code compiled clean, passed every fixed-base robot, and then wrote past the end of the arena on
go2-FLOATING only -- taking the whole tier binary down with it (one gpuErrchk exit() kills every algo
in the batch process, so a single wrong `max` silently cost us an entire cell's timing data).

An under-sized arena also DEFEATS the fits-check: `select_shared_tier_3way` asks "does this arena fit in
the smem budget?" using the same under-counted number, so it happily picks TIER_SHARED for a kernel that
does not actually fit.

WHAT THIS CHECKS. Purely static, on the GENERATED header -- no compile, no GPU, no blast radius on
codegen. `gen_declare_shared_arena` already emits every carve as a `// GRID shared arena layout` comment
block listing each region and its element count. So for every __global__ kernel we sum the regions it
actually carves and assert the macro that sizes its launch is >= that sum, at EVERY tier.

The invariant is `>=`, not `==`: a spill ladder legitimately leaves slack (the arena is a max over rungs,
and the picked rung may not be the argmax). Slack is waste, not corruption. UNDER-counting is corruption.
"""
from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (robot, floating) -- the matrix that actually exercises the spill ladders. go2-floating is the cell
# that §1t blew up on; iiwa14 is the fixed-base control; fr3 carries the mimic fold.
_CASES = [("iiwa14", False), ("go2", True), ("go2", False), ("fr3", False)]

# Regions the arena carves but that are NOT part of the T-count the macro reports (the macro accounts
# for them separately, via its own topology/linalg arguments).
_NON_T_REGION = re.compile(r"^//\s+(int|bytes)\s")
_T_REGION = re.compile(r"^//\s+T\s+(\w+)\[([^\]]+)\]")
_ARENA_START = re.compile(r"^//\s*GRID shared arena layout\s*$")
_KERNEL_DEF = re.compile(r"\b__global__\b")
_FUNC_NAME = re.compile(r"\bvoid\s+(\w+)\s*\(")
# grid_shared_arena_bytes<T>(<T_COUNT>, TOPOLOGY_HELPERS_COUNT, <linalg bytes>)
_MACRO_DEF = re.compile(r"\b(\w+)_DYNAMIC_SHARED_MEM_BYTES\s*\(\)")
_ARENA_BYTES_CALL = re.compile(r"grid_shared_arena_bytes<T>\(\s*([0-9]+)\s*,")


def _generate(robot_id: str, floating: bool, out: Path) -> Path:
    from URDFParser import URDFParser
    from GRiDCodeGenerator import GRiDCodeGenerator
    urdf = REPO_ROOT / "robot_assets" / f"{robot_id}.urdf"
    if not urdf.exists():
        pytest.skip(f"{robot_id}.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
        if robot is None:
            pytest.skip(f"{robot_id} URDF parse failed")
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            codegen_profile="all", output_path=str(out))
    return out


def _macro_tier_counts(text: str) -> dict[str, list[int]]:
    """ALGO -> [T-counts, one per tier branch] from the *_DYNAMIC_SHARED_MEM_BYTES definitions.

    A non-tiered macro has a single grid_shared_arena_bytes call; a tiered one has three (SHARED /
    LITE / MINIMAL). We keep them all and later require the macro to cover the carve at EVERY tier
    where the carve applies -- an under-count on ONE tier is exactly the §1t failure mode.
    """
    out: dict[str, list[int]] = {}
    for line in text.splitlines():
        m = _MACRO_DEF.search(line)
        if not m or "return" not in line:
            continue
        counts = [int(c) for c in _ARENA_BYTES_CALL.findall(line)]
        if counts:
            out[m.group(1)] = counts
    return out


_TIER_BRANCH = re.compile(r"RESOURCE_TIER\s*==\s*TIER_(SHARED|LITE|MINIMAL)")
_TIER_IDX = {"SHARED": 0, "LITE": 1, "MINIMAL": 2}


def _kernel_carves(text: str) -> list[tuple[str, int | None, int, int]]:
    """[(kernel_name, tier_idx_or_None, carve_with_tiered_slot, carve_without_tiered_slot)].

    A kernel emits ONE arena block PER TIER BRANCH (gen_tier_dispatch fans out
    `if constexpr (RESOURCE_TIER == TIER_SHARED) {...} else if (...TIER_LITE) {...} else {...}`), and the
    block is the first thing inside its branch -- so the branch opener is within a couple of lines above
    it. `tier_idx=None` means the kernel has a single, non-tier-dispatched body (the 3 picks agreed and
    gen_tier_dispatch collapsed them), which must then be covered at EVERY tier.

    A slot marked `(TIER_SHARED only; ...)` is carved from the arena only at TIER_SHARED and routed to the
    global workspace otherwise -- it belongs to the first figure, not the second. Non-integer counts (a
    C++ constexpr slot expression) are not statically summable; we skip those blocks rather than guess.
    """
    lines = text.splitlines()
    carves: list[tuple[str, int | None, int, int]] = []
    for i, line in enumerate(lines):
        if not _ARENA_START.match(line.strip()):
            continue
        # Which tier branch is this block in? The opener is 1-3 lines up (the block leads the branch).
        tier: int | None = None
        for j in range(i - 1, max(-1, i - 4), -1):
            bm = _TIER_BRANCH.search(lines[j])
            if bm:
                tier = _TIER_IDX[bm.group(1)]
                break
            if lines[j].strip().startswith("else"):
                tier = 2          # bare `else` closing a 3-way tier dispatch == the MINIMAL fallback
                break
        # Walk back to the enclosing signature. `void foo(` is the signature line; `__global__` sits a
        # couple of lines ABOVE it (after the template line), so keep scanning past the name.
        name, is_kernel, name_line = None, False, None
        for j in range(i - 1, max(-1, i - 60), -1):
            fm = _FUNC_NAME.search(lines[j])
            if fm and name is None:
                name, name_line = fm.group(1), j
            if name is not None and _KERNEL_DEF.search(lines[j]):
                is_kernel = True
                break
            if name is not None and name_line is not None and j < name_line - 4:
                break             # searched the decl specifiers above the signature; not a __global__
        if not is_kernel or name is None:
            continue
        with_ct, without_ct, ok = 0, 0, True
        for k in range(i + 1, len(lines)):
            s = lines[k].strip()
            if _NON_T_REGION.match(s):
                continue          # int/bytes regions: the macro accounts for these via its own args
            tm = _T_REGION.match(s)
            if not tm:
                break             # end of the comment block
            try:
                n = int(tm.group(2))
            except ValueError:
                ok = False        # constexpr slot expression -- not statically summable
                break
            with_ct += n
            if "TIER_SHARED only" not in s:
                without_ct += n
        if ok:
            carves.append((name, tier, with_ct, without_ct))
    return carves


@pytest.mark.parametrize(("robot_id", "floating"), _CASES)
def test_shared_mem_macro_covers_kernel_carve(robot_id, floating, tmp_path):
    header = _generate(robot_id, floating, tmp_path / "grid.cuh")
    text = header.read_text()
    macros = _macro_tier_counts(text)
    carves = _kernel_carves(text)
    assert carves, "parsed no kernel arena blocks -- the emitter's comment format changed"
    assert macros, "parsed no *_DYNAMIC_SHARED_MEM_BYTES definitions"

    violations, checked = [], 0
    for kname, tier, with_ct, without_ct in carves:
        # kernel `foo_kernel` / `foo_kernel_single_timing` is sized by FOO_DYNAMIC_SHARED_MEM_BYTES
        stem = kname.replace("_single_timing", "")
        stem = stem[:-len("_kernel")] if stem.endswith("_kernel") else stem
        tiers = macros.get(stem.upper())
        if tiers is None:
            continue                            # kernel with no macro of its own (shares a sibling's)
        # Which macro tier(s) must cover this block, and what does the block carve at each?
        # A TIER_SHARED-only slot is in the arena at tier 0 and routed to global memory elsewhere.
        if tier is None:                        # collapsed body: must be covered at EVERY tier
            todo = [(t, with_ct if t == 0 else without_ct) for t in range(len(tiers))]
        else:
            todo = [(min(tier, len(tiers) - 1), with_ct if tier == 0 else without_ct)]
        for t, want in todo:
            checked += 1
            have = tiers[t]
            if have < want:
                violations.append(
                    f"{kname}: tier {t} macro {stem.upper()}_DYNAMIC_SHARED_MEM_BYTES reports "
                    f"{have} T-elements but the kernel CARVES {want} (short by {want - have}) "
                    f"-- arena under-count, see debugging-guide §1t")

    assert checked >= 20, f"only matched {checked} kernel/tier pairs -- the naming convention drifted"
    assert not violations, (
        f"{robot_id} ({'floating' if floating else 'fixed'}): shared arena under-counted for "
        f"{len(violations)} kernel/tier(s):\n  " + "\n  ".join(violations))

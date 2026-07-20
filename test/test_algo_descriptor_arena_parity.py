"""Parity + invariant net for the per-algo DESCRIPTOR arena composer (item M, Step 3).

Descriptor-table Step 3 folded the ~272 hand-written `*_t_count` arena expressions in
`GRiDCodeGenerator.gen_add_constants_helpers` into the table: every
`select_shared_tier_3way` site is now DRIVEN from `compose_arena_rungs` /
`compose_arena_full` in `algo_registry.py` (docs/open-tasks/design_descriptor_table_spec.md
§3-4). Step 3.6 deleted the inline `assert composed == legacy` shims + the `_*_legacy`
imperative duplicates — the composer is now the single source of arena logic.

★★ REMOVED 2026-07-14 — `test_arena_full_composer_matches_generator` WAS CIRCULAR AND IS GONE.

It asserted `compose_arena_full(k, ctx) == gen._arena_full_t_counts[k]`. But after the Step-3 fold, 22
of the 34 entries in that snapshot are THEMSELVES assigned from the composer — e.g.
`GRiDCodeGenerator.py` has `"fdsva_so": _fdsva_so_arenas[0]` where
`_fdsva_so_arenas = compose_arena_rungs("fdsva_so", self._arena_ctx)`. So for those keys the test
asserted `compose_arena_full(k) == compose_arena_full(k)`. **It could not fail.**

That is not a hypothetical. It is exactly what happened: the parity net reported GREEN for the entire
life of the §1t bug, while `fdsva_so`'s arena under-counted by 1077 elements and wrote past the end of
shared memory on go2-floating. The fold deleted the `assert composed == legacy` shims but rewired the
"truth" side of the test to the code under test. A green test over an unverified arena is WORSE than no
test — it launders the thing it was built to catch.

★ THE ARENA'S REAL AUTHORITY IS NOW `test/test_shared_arena_covers_carve.py`. That one is genuinely
INDEPENDENT: it compares each kernel's launch-sizing macro against the regions that kernel ACTUALLY
CARVES (parsed out of the emitted header), i.e. two paths that are computed separately and must agree.
Its positive control — re-introducing the bad rung — names `fdsva_so tier 0 short-by-1077` immediately.
**Do not "restore" a composer-vs-generator comparison here. Any such test is a tautology by construction,
because the composer is now the ONLY source of arena logic.**

What remains here are the three checks that are NOT circular — each compares the composer against
something computed independently of it:

  * `test_arena_ladder_invariants` — every composed rung is positive and rung[0] == the full arena
    (structural properties, not a self-comparison).
  * `test_rt_reservation` — the §2 bug-class guard, and the strongest test in this file: it is a
    DIFFERENTIAL check. Compose with `runtime_transform` OFF and ON; every s_temp-domain full arena must
    grow by EXACTLY 36*NJ and every other by 0. The composer cannot satisfy this by agreeing with itself.
  * `test_composed_keys_exclude_only_so_dispatch` — the composed set is exactly the generator's arena
    keys minus the one deferred full (idsva_so_body_frame). A bijection, not a value check.

Matrix (orthogonal axes): iiwa14-fixed (T-only, n==nv), go2-floating (base-DOF terms,
n>nv), fr3-fixed (mimic, NB>nv), + iiwa14-fixed with runtime_transform (the rt_xfixed
term, the §2 bug class). `idsva_so_body_frame`'s FULL stays deferred (its floating path is
a grav_full_spill picker override that is not ctx-pure); its fixed-base rung ladder IS
composed. Pure Python — no nvcc, no GPU.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algo_registry import (
    ARENA_COMPOSED_KEYS,
    ARENA_RUNG_KEYS,
    arena_ctx_from_codegen,
    compose_arena_full,
    compose_arena_rungs,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

# (robot_id, base_mode, runtime_transform). Small robots only (pure-Python codegen).
_MATRIX = [
    ("iiwa14", "fixed", False),     # T-only fixed base (n==nv)
    ("go2", "floating", False),     # floating: base-DOF arena terms (n>nv)
    ("fr3", "fixed", False),        # mimic: NB > nv arena terms (18*NJ vaf bands)
    ("iiwa14", "fixed", True),      # runtime_transform: the rt_xfixed reservation (§2)
]

# idsva_so_body_frame's FULL is not composed (floating grav_full_spill picker override,
# not ctx-pure). Its fixed-base rung ladder IS composed and checked via the rung net.
_SO_DISPATCH_DEFERRED = {"idsva_so_body_frame"}

# §2 guard: the full arenas that reserve the rt_xfixed band under runtime_transform (a
# 36*NJ region on the load_update_XImats helper). Frozen classification — a composer edit
# that drops rt from one of these (the §2 silent under-size) OR adds it to a NO-RT algo
# flips membership and fails test_rt_reservation. Computed once from the composer; the
# rt-domain rarely changes, so this set is stable (unlike exact arena values).
_RT_RESERVING_FULL_KEYS = frozenset({
    "aba", "coriolis_matrix", "crba", "f_ext_gradient", "f_ext_gradient_dq", "fdsva_so", "forward_dynamics",
    "forward_dynamics_gradient", "forward_dynamics_parameter_gradient", "generalized_gravity",
    "idsva_so_world_frame", "integrator", "integrator_gradient", "integrator_hessian",
    "integrator_with_gradient", "inverse_dynamics", "inverse_dynamics_gradient",
    "inverse_dynamics_regressor", "kinetic_energy_regressor", "minv", "nonlinear_effects",
})


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _codegen_for(robot_id, base_mode, runtime_transform, tmp_path):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except Exception as exc:  # missing robot_descriptions asset, etc.
        pytest.skip(f"cannot resolve {robot_id}-{base_mode}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid"
    )
    header = tmp_path / "grid.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # "dynamics-core" is the leanest valid profile; gen_add_constants_helpers computes
        # ALL arena locals + the descriptor dicts unconditionally regardless of profile, so
        # this is fast (no SO/kinematics kernel emission) yet complete. Generation also runs
        # the in-line arena invariant self-check (positive rungs, rung[0]==full) on this robot.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            codegen_profile="dynamics-core",
            runtime_transform=runtime_transform,
        )
    return codegen


def test_no_circular_composer_vs_generator_check_is_reintroduced():
    """REGRESSION GUARD for the removed tautology (see the module docstring).

    `GRiDCodeGenerator._arena_full_t_counts` is now POPULATED BY THE COMPOSER — e.g.
    `"fdsva_so": _fdsva_so_arenas[0]` where `_fdsva_so_arenas = compose_arena_rungs("fdsva_so", ctx)`.
    Comparing `compose_arena_full(k)` against that snapshot therefore compares the composer to itself.
    The old `test_arena_full_composer_matches_generator` did exactly that and stayed GREEN through the
    entire life of the §1t shared-memory OOB.

    This guard fails if someone re-adds such a comparison. The tautology's signature is: a single
    function that BOTH calls a `compose_arena_*` composer AND reads a `_arena_*_t_counts` generator
    snapshot — i.e. it compares the composer to a value the composer produced. Reading only the snapshot's
    KEYS (as the bijection test does) is fine and does not trip this; calling only the composer (the
    invariant/rt/carve tests) is fine too. It is the CONJUNCTION that is circular. AST-based, so
    docstrings/comments don't count.
    """
    import ast
    COMPOSERS = {"compose_arena_full", "compose_arena_rungs"}
    SNAPSHOTS = {"_arena_full_t_counts", "_arena_rung_t_counts"}
    src = (Path(__file__).parent / "test_algo_descriptor_arena_parity.py").read_text()
    offenders = []
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "test_no_circular_composer_vs_generator_check_is_reintroduced":
            continue
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        attrs = {a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)}
        if (names & COMPOSERS) and ((names | attrs) & SNAPSHOTS):
            offenders.append(fn.name)
    assert not offenders, (
        f"{offenders}: a composer-vs-generator-snapshot comparison was re-added. The `_arena_*_t_counts` "
        "snapshots are populated BY the composer, so comparing compose_arena_*() against them is a "
        "tautology that CANNOT FAIL — it is exactly how §1t reached production. The arena's independent "
        "authority is test/test_shared_arena_covers_carve.py (launch macro vs the kernel's ACTUAL carve)."
    )


# NOTE: the former `test_arena_ladder_matches_generator_record` was ALSO removed 2026-07-14. It compared
# `compose_arena_rungs(k)` against the generator's `_arena_rung_t_counts[k]` — but that record is ALSO
# composer-sourced (Step 3.4), so it was the same tautology on the rung ladders. Its only non-circular
# content (ladder length / key composed-but-not-captured) is already covered by test_arena_ladder_invariants
# and test_composed_keys_exclude_only_so_dispatch. The regression guard below forbids re-adding either
# form (full OR rung).


@pytest.mark.parametrize("robot_id,base_mode,runtime_transform", _MATRIX)
def test_arena_ladder_invariants(robot_id, base_mode, runtime_transform, tmp_path):
    """Structural invariants on every composed ladder (mirrors the generator's in-line
    self-check, on the matrix): all rungs strictly positive; rung[0] == the full arena for
    keys whose full is composed. Catches a future closure edit that produces a
    negative/absurd/inconsistent arena (the §2 under-size class), robot-agnostically."""
    gen = _codegen_for(robot_id, base_mode, runtime_transform, tmp_path)
    ctx = gen._arena_ctx
    bad = []
    for key in sorted(ARENA_RUNG_KEYS):
        rungs = compose_arena_rungs(key, ctx)
        if not all(r > 0 for r in rungs):
            bad.append(f"  {key}: non-positive rung in {rungs}")
        if key in ARENA_COMPOSED_KEYS:
            full = compose_arena_full(key, ctx)
            if rungs[0] != full:
                bad.append(f"  {key}: rung[0]={rungs[0]} != full={full}")
    for key in sorted(ARENA_COMPOSED_KEYS):
        if compose_arena_full(key, ctx) <= 0:
            bad.append(f"  {key}: non-positive full arena")
    assert not bad, (
        f"arena ladder invariants violated on {robot_id}-{base_mode} "
        f"(runtime_transform={runtime_transform}):\n" + "\n".join(bad)
    )


@pytest.mark.parametrize("robot_id,base_mode", [("iiwa14", "fixed"), ("go2", "floating"), ("fr3", "fixed")])
def test_rt_reservation(robot_id, base_mode, tmp_path):
    """§2 bug-class guard. Under runtime_transform each s_temp-domain full arena must grow
    by EXACTLY 36*NJ (the rt_xfixed reservation on the load_update_XImats helper) and every
    other by 0 — never a fraction, never double. Membership in the rt-reserving set is
    frozen: a composer edit that silently drops rt from an s_temp algo (the §2 under-size)
    or adds it to a kinematics algo flips the delta and fails here. Independent of the
    generator's runtime_transform flag (builds explicit rt / no-rt ctx variants)."""
    gen = _codegen_for(robot_id, base_mode, False, tmp_path)
    nj = gen.robot.get_num_joints()
    rt = 36 * nj
    ctx_rt = arena_ctx_from_codegen(gen, rt=rt)
    ctx_no = arena_ctx_from_codegen(gen, rt=0)
    bad = []
    for key in sorted(ARENA_COMPOSED_KEYS):
        delta = compose_arena_full(key, ctx_rt) - compose_arena_full(key, ctx_no)
        expected = rt if key in _RT_RESERVING_FULL_KEYS else 0
        if delta != expected:
            bad.append(f"  {key}: rt-delta={delta} expected={expected} (rt_reserving={key in _RT_RESERVING_FULL_KEYS})")
    assert not bad, (
        f"rt_xfixed reservation drift on {robot_id}-{base_mode} (NJ={nj}, rt={rt}):\n"
        + "\n".join(bad)
        + "\n(a nonzero-but-wrong delta = §2 silent under/over-size; update _RT_RESERVING_FULL_KEYS"
          " only for a DELIBERATE domain change.)"
    )


def test_composed_keys_exclude_only_so_dispatch(tmp_path):
    """The composed-full set is exactly the generator's arena keys minus the one deferred
    full (idsva_so_body_frame — floating grav_full_spill picker override). Guards against
    silently dropping an algo from the net or forgetting to compose one."""
    gen = _codegen_for("iiwa14", "fixed", False, tmp_path)
    arena_keys = set(gen._arena_full_t_counts)
    assert ARENA_COMPOSED_KEYS == arena_keys - _SO_DISPATCH_DEFERRED, (
        "composed-key set drifted from (generator arena keys - deferred).\n"
        f"  composed-only: {sorted(ARENA_COMPOSED_KEYS - (arena_keys - _SO_DISPATCH_DEFERRED))}\n"
        f"  arena-only:    {sorted((arena_keys - _SO_DISPATCH_DEFERRED) - ARENA_COMPOSED_KEYS)}"
    )

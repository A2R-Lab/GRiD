"""Parity + invariant net for the per-algo DESCRIPTOR arena composer (item M, Step 3).

Descriptor-table Step 3 folded the ~272 hand-written `*_t_count` arena expressions in
`GRiDCodeGenerator.gen_add_constants_helpers` into the table: every
`select_shared_tier_3way` site is now DRIVEN from `compose_arena_rungs` /
`compose_arena_full` in `algo_registry.py` (docs/open-tasks/design_descriptor_table_spec.md
§3-4). Step 3.6 deleted the inline `assert composed == legacy` shims + the `_*_legacy`
imperative duplicates — the composer is now the single source of arena logic.

This test is the regression net that replaced those shims (user choice: invariants +
rt-guard, not an exact-value golden — arenas are actively tuned by the surgical-spill
perf work, so pinning exact values would churn):

  * `test_arena_full_composer_matches_generator` — the composer reproduces the
    generator's own `_arena_full_t_counts` snapshot. Several of those entries are still
    the surviving IMPERATIVE full locals (they feed the `*_MAX_SHARED_MEM_COUNT` macros),
    so this remains a real composer-vs-imperative cross-check for those algos; for the
    rest it is a self-consistency check.
  * `test_arena_ladder_invariants` — every composed rung is positive and rung[0] == the
    full arena (the generator asserts this in-line on EVERY robot; here on the matrix).
  * `test_rt_reservation` — the §2 bug-class guard: under `runtime_transform` each
    s_temp-domain full arena grows by EXACTLY 36*NJ (the rt_xfixed reservation) and every
    other by 0. A dropped/duplicated rt term (the §2 silent under-size) fails this.
  * `test_composed_keys_exclude_only_so_dispatch` — the composed set is exactly the
    generator's arena keys minus the one deferred full (idsva_so_body_frame).

Matrix (orthogonal axes): iiwa14-fixed (T-only, n==nv), go2-floating (base-DOF terms,
n>nv), fr3-fixed (mimic, NB>nv), + iiwa14-fixed with runtime_transform (the rt_xfixed
term, the §2 bug class). `idsva_so_body_frame`'s FULL stays deferred (its floating path is
a grav_full_spill picker override that is not ctx-pure); its fixed-base rung ladder IS
composed. Pure Python — no nvcc, no GPU.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algo_registry import (
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
    "aba", "coriolis_matrix", "crba", "f_ext_gradient", "fdsva_so", "forward_dynamics",
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


@pytest.mark.parametrize("robot_id,base_mode,runtime_transform", _MATRIX)
def test_arena_full_composer_matches_generator(robot_id, base_mode, runtime_transform, tmp_path):
    """The composer reproduces the generator's `_arena_full_t_counts` snapshot on each
    matrix robot. For algos whose imperative full local survives (it feeds the
    `*_MAX_SHARED_MEM_COUNT` macros) this is a real composer-vs-imperative check; for the
    rest it is self-consistency. Also checks the composed rung ladders vs the generator's
    `_arena_rung_t_counts` record + rung[0]==full."""
    gen = _codegen_for(robot_id, base_mode, runtime_transform, tmp_path)
    ctx = gen._arena_ctx   # the exact snapshot generation used to drive the folded arenas
    truth = gen._arena_full_t_counts

    mismatches = []
    for key in sorted(ARENA_COMPOSED_KEYS):
        assert key in truth, f"{key} composed but absent from _arena_full_t_counts"
        composed = compose_arena_full(key, ctx)
        if composed != truth[key]:
            mismatches.append(f"  {key}: composer={composed} generator={truth[key]} (Δ={composed - truth[key]})")
    assert not mismatches, (
        f"arena_full composer disagrees with the generator on "
        f"{robot_id}-{base_mode} (runtime_transform={runtime_transform}):\n"
        + "\n".join(mismatches)
    )

    rung_truth = getattr(gen, "_arena_rung_t_counts", {})
    rung_mismatches = []
    for key in sorted(ARENA_RUNG_KEYS):
        rungs = compose_arena_rungs(key, ctx)
        if key in rung_truth and tuple(rungs) != tuple(rung_truth[key]):
            rung_mismatches.append(f"  {key}: composer={tuple(rungs)} generator={tuple(rung_truth[key])}")
    assert not rung_mismatches, (
        f"arena rung composer disagrees on {robot_id}-{base_mode} "
        f"(runtime_transform={runtime_transform}):\n" + "\n".join(rung_mismatches)
    )


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

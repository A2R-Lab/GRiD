"""Step-3.0 parity net for the per-algo DESCRIPTOR arena composer (item M).

Descriptor-table Step 3 folds the ~272 hand-written `*_t_count` arena expressions in
`GRiDCodeGenerator.gen_add_constants_helpers` into the table (design:
docs/open-tasks/design_descriptor_table_spec.md §3-4). It is the RISKIEST step — a
wrong `arena_fn` silently under-sizes shared memory (the §2 rt_xfixed bug class) —
so it lands one algo per commit behind a byte-diff + CUDA-equivalence gate.

Step 3.0 GENERATES NOTHING. It lands the `ArenaCtx` snapshot + the per-algo
`arena_full_fn` closures in `algo_registry.py`, DECOUPLED from the imperative arena
math, and this test asserts each closure reproduces the imperative
`GRiDCodeGenerator._arena_full_t_counts[key]` (the FULL / least-spill / rung-0 arena)
on every matrix robot. That is the safety net later commits lean on: once the
composer provably agrees, generation can be driven from the table under a
byte-identical gate. This is pure Python — no nvcc, no GPU.

Matrix (orthogonal axes): iiwa14-fixed (T-only), go2-floating (base-DOF terms),
fr3-fixed (mimic, NB>nv), + iiwa14-fixed with runtime_transform (the rt_xfixed term,
the §2 bug class). fdsva_so (8-rung ladder) is composed as of 3.4. The 2 idsva_so
SO-dispatch algos (body/world) have per_base_override / workspace closures inseparable
from their multi-rung emission and are composed in commit 3.5 — captured in
`_arena_full_t_counts` but excluded from `ARENA_COMPOSED_KEYS` (asserted below).
"""

from __future__ import annotations

import contextlib
import os

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algo_registry import (
    ARENA_COMPOSED_KEYS,
    ARENA_RUNG_KEYS,
    compose_arena_full,
    compose_arena_rungs,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

# (robot_id, base_mode, runtime_transform). Small robots only (pure-Python codegen).
_MATRIX = [
    ("iiwa14", "fixed", False),     # T-only fixed base
    ("go2", "floating", False),     # floating: base-DOF arena terms
    ("fr3", "fixed", False),        # mimic: NB > nv arena terms (18*NJ vaf bands)
    ("iiwa14", "fixed", True),      # runtime_transform: the rt_xfixed reservation (§2)
]

# Captured in _arena_full_t_counts but composed later (3.5), so not asserted here.
# fdsva_so was folded in 3.4 (now composed); idsva_so body/world remain deferred.
_SO_DISPATCH_DEFERRED = {"idsva_so_body_frame", "idsva_so_world_frame"}


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
        # "dynamics-core" is the leanest valid profile; gen_add_constants_helpers
        # computes ALL arena locals + _arena_full_t_counts unconditionally regardless
        # of profile, so this is fast (no SO/kinematics kernel emission) yet complete.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            codegen_profile="dynamics-core",
            runtime_transform=runtime_transform,
        )
    return codegen


@pytest.mark.parametrize("robot_id,base_mode,runtime_transform", _MATRIX)
def test_arena_full_composer_matches_imperative(robot_id, base_mode, runtime_transform, tmp_path):
    """Every composed algo's `arena_full_fn(ctx)` reproduces the imperative
    FULL/rung-0 arena t_count exactly, on each matrix robot (incl. runtime_transform)."""
    gen = _codegen_for(robot_id, base_mode, runtime_transform, tmp_path)
    truth = gen._arena_full_t_counts
    ctx = gen._arena_ctx   # the exact snapshot generation used to drive the folded arenas

    mismatches = []
    for key in sorted(ARENA_COMPOSED_KEYS):
        assert key in truth, f"{key} composed but absent from _arena_full_t_counts"
        composed = compose_arena_full(key, ctx)
        if composed != truth[key]:
            mismatches.append(f"  {key}: composer={composed} imperative={truth[key]} (Δ={composed - truth[key]})")
    assert not mismatches, (
        f"arena_full composer disagrees with imperative t_count on "
        f"{robot_id}-{base_mode} (runtime_transform={runtime_transform}):\n"
        + "\n".join(mismatches)
    )

    # Laddered folds (3.2+): the composed rung arenas reproduce the imperative rungs,
    # and rung[0] (least-spill) equals the FULL arena (self-consistency).
    rung_truth = getattr(gen, "_arena_rung_t_counts", {})
    rung_mismatches = []
    for key in sorted(ARENA_RUNG_KEYS):
        rungs = compose_arena_rungs(key, ctx)
        if key in rung_truth and tuple(rungs) != tuple(rung_truth[key]):
            rung_mismatches.append(f"  {key}: composer={tuple(rungs)} imperative={tuple(rung_truth[key])}")
        if key in ARENA_COMPOSED_KEYS and rungs[0] != compose_arena_full(key, ctx):
            rung_mismatches.append(f"  {key}: rung[0]={rungs[0]} != full={compose_arena_full(key, ctx)}")
    assert not rung_mismatches, (
        f"arena rung composer disagrees on {robot_id}-{base_mode} "
        f"(runtime_transform={runtime_transform}):\n" + "\n".join(rung_mismatches)
    )


def test_composed_keys_exclude_only_so_dispatch(tmp_path):
    """The composed set is exactly the imperative arena keys minus the 3 SO-dispatch
    algos (which land in 3.4/3.5). Guards against silently dropping an algo from the
    net or forgetting to compose one when its fold lands."""
    gen = _codegen_for("iiwa14", "fixed", False, tmp_path)
    arena_keys = set(gen._arena_full_t_counts)
    # integrator_hessian is captured for the plant_step_hessian fold but has no
    # standalone benchmarked kernel (has_kernel_attr False) and is not a rung-0 arena
    # the composer owns yet — treat it like the SO-dispatch deferrals.
    deferred = _SO_DISPATCH_DEFERRED | {"integrator_hessian"}
    assert ARENA_COMPOSED_KEYS == arena_keys - deferred, (
        "composed-key set drifted from (imperative arena keys - deferred).\n"
        f"  composed-only: {sorted(ARENA_COMPOSED_KEYS - (arena_keys - deferred))}\n"
        f"  arena-only:    {sorted((arena_keys - deferred) - ARENA_COMPOSED_KEYS)}"
    )
